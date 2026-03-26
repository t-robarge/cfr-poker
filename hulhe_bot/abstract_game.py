from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .bucketing import Bucketer
from .cards import compare_hands, shuffled_deck, stable_seed
from .config import AbstractHULHEConfig
from .models import Action, Street


def pair_key(left: int, right: int) -> str:
    return f"{left}-{right}"


@dataclass(frozen=True, slots=True)
class PublicTreeState:
    street: Street
    current_player: int
    button: int
    pot: int
    total_contrib: tuple[int, int]
    street_contrib: tuple[int, int]
    current_to_call: int
    bet_level: int
    acted_this_street: tuple[int, ...]
    last_aggressor: int | None
    history_by_street: tuple[str, str, str, str]
    terminal_type: str | None = None
    folded_player: int | None = None

    def history_id(self) -> str:
        labels = []
        for street, sequence in zip(Street, self.history_by_street):
            if sequence:
                labels.append(f"{street.value[0]}:{sequence}")
        return "/".join(labels) or "root"


@dataclass(slots=True)
class AbstractGameSpec:
    format_name: str
    root_node: str
    config: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    initial_bucket_distribution: dict[str, float]
    street_transitions: dict[str, dict[str, dict[str, float]]]
    river_showdown_share: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "root_node": self.root_node,
            "config": self.config,
            "nodes": self.nodes,
            "initial_bucket_distribution": self.initial_bucket_distribution,
            "street_transitions": self.street_transitions,
            "river_showdown_share": self.river_showdown_share,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> "AbstractGameSpec":
        payload = json.loads(Path(path).read_text())
        return cls(
            format_name=payload["format_name"],
            root_node=payload["root_node"],
            config=payload["config"],
            nodes=payload["nodes"],
            initial_bucket_distribution=payload["initial_bucket_distribution"],
            street_transitions=payload["street_transitions"],
            river_showdown_share=payload["river_showdown_share"],
            metadata=payload.get("metadata", {}),
        )


class AbstractGameBuilder:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.bucketer = Bucketer(self.config)

    def build(self, config: AbstractHULHEConfig | None = None) -> str:
        cfg = config or self.config
        spec = self.build_spec(cfg)
        return spec.save(cfg.abstraction_file)

    def build_spec(self, config: AbstractHULHEConfig | None = None) -> AbstractGameSpec:
        cfg = config or self.config
        nodes = self._build_betting_tree(cfg)
        initial_distribution, transitions, showdown = self._sample_abstraction_tables(cfg)
        return AbstractGameSpec(
            format_name="hulhe_abstract_game_v1",
            root_node="root",
            config=cfg.to_dict(),
            nodes=nodes,
            initial_bucket_distribution=initial_distribution,
            street_transitions=transitions,
            river_showdown_share=showdown,
            metadata={
                "node_count": len(nodes),
                "street_transition_sources": {
                    street: len(mapping) for street, mapping in transitions.items()
                },
            },
        )

    def _build_betting_tree(self, config: AbstractHULHEConfig) -> dict[str, dict[str, Any]]:
        root = PublicTreeState(
            street=Street.PREFLOP,
            current_player=0,
            button=0,
            pot=config.small_blind + config.big_blind,
            total_contrib=(config.small_blind, config.big_blind),
            street_contrib=(config.small_blind, config.big_blind),
            current_to_call=config.big_blind,
            bet_level=1,
            acted_this_street=(),
            last_aggressor=None,
            history_by_street=("", "", "", ""),
        )
        nodes: dict[str, dict[str, Any]] = {}
        self._visit_public_state(root, config, nodes)
        return nodes

    def _visit_public_state(
        self,
        state: PublicTreeState,
        config: AbstractHULHEConfig,
        nodes: dict[str, dict[str, Any]],
    ) -> None:
        node_id = state.history_id()
        if node_id in nodes:
            return
        if state.terminal_type is not None:
            nodes[node_id] = {
                "node_id": node_id,
                "street": state.street.value,
                "current_player": state.current_player,
                "history_id": node_id,
                "legal_actions": [],
                "pot": state.pot,
                "total_contrib": list(state.total_contrib),
                "street_contrib": list(state.street_contrib),
                "current_to_call": state.current_to_call,
                "bet_level": state.bet_level,
                "terminal_type": state.terminal_type,
                "folded_player": state.folded_player,
                "transitions": {},
            }
            return

        legal_actions = self._legal_actions(state, config)
        transitions: dict[str, dict[str, Any]] = {}
        nodes[node_id] = {
            "node_id": node_id,
            "street": state.street.value,
            "current_player": state.current_player,
            "history_id": node_id,
            "legal_actions": [action.value for action in legal_actions],
            "pot": state.pot,
            "total_contrib": list(state.total_contrib),
            "street_contrib": list(state.street_contrib),
            "current_to_call": state.current_to_call,
            "bet_level": state.bet_level,
            "terminal_type": None,
            "folded_player": None,
            "transitions": transitions,
        }
        for action in legal_actions:
            next_state, chance_stage = self._apply_public_action(state, action, config)
            transitions[action.value] = {
                "next_node": next_state.history_id(),
                "chance_stage": chance_stage,
            }
            self._visit_public_state(next_state, config, nodes)

    def _legal_actions(self, state: PublicTreeState, config: AbstractHULHEConfig) -> tuple[Action, ...]:
        to_call = state.current_to_call - state.street_contrib[state.current_player]
        actions: list[Action] = [Action.FOLD] if to_call > 0 else []
        actions.append(Action.CALL if to_call > 0 else Action.CHECK)
        if state.bet_level < config.bet_cap:
            actions.append(Action.RAISE)
        return tuple(actions)

    def _apply_public_action(
        self,
        state: PublicTreeState,
        action: Action,
        config: AbstractHULHEConfig,
    ) -> tuple[PublicTreeState, str | None]:
        player = state.current_player
        opponent = 1 - player
        to_call = state.current_to_call - state.street_contrib[player]
        total_contrib = list(state.total_contrib)
        street_contrib = list(state.street_contrib)
        history_by_street = list(state.history_by_street)
        street_index = list(Street).index(state.street)
        acted = set(state.acted_this_street)
        pot = state.pot
        current_to_call = state.current_to_call
        bet_level = state.bet_level
        last_aggressor = state.last_aggressor
        code = {
            Action.FOLD: "f",
            Action.CHECK: "k",
            Action.CALL: "c",
            Action.RAISE: "r",
        }[action]

        if action == Action.FOLD:
            history_by_street[street_index] += code
            return (
                PublicTreeState(
                    street=state.street,
                    current_player=opponent,
                    button=state.button,
                    pot=pot,
                    total_contrib=tuple(total_contrib),
                    street_contrib=tuple(street_contrib),
                    current_to_call=current_to_call,
                    bet_level=bet_level,
                    acted_this_street=tuple(sorted(acted)),
                    last_aggressor=last_aggressor,
                    history_by_street=tuple(history_by_street),
                    terminal_type="fold",
                    folded_player=player,
                ),
                None,
            )

        if action == Action.CHECK:
            history_by_street[street_index] += code
            acted.add(player)
            if len(acted) == 2:
                return self._advance_public_street(
                    PublicTreeState(
                        street=state.street,
                        current_player=player,
                        button=state.button,
                        pot=pot,
                        total_contrib=tuple(total_contrib),
                        street_contrib=tuple(street_contrib),
                        current_to_call=current_to_call,
                        bet_level=bet_level,
                        acted_this_street=tuple(sorted(acted)),
                        last_aggressor=last_aggressor,
                        history_by_street=tuple(history_by_street),
                    )
                )
            return (
                PublicTreeState(
                    street=state.street,
                    current_player=opponent,
                    button=state.button,
                    pot=pot,
                    total_contrib=tuple(total_contrib),
                    street_contrib=tuple(street_contrib),
                    current_to_call=current_to_call,
                    bet_level=bet_level,
                    acted_this_street=tuple(sorted(acted)),
                    last_aggressor=last_aggressor,
                    history_by_street=tuple(history_by_street),
                ),
                None,
            )

        if action == Action.CALL:
            total_contrib[player] += to_call
            street_contrib[player] += to_call
            pot += to_call
            history_by_street[street_index] += code
            acted.add(player)
            if state.last_aggressor is not None or len(acted) == 2:
                return self._advance_public_street(
                    PublicTreeState(
                        street=state.street,
                        current_player=player,
                        button=state.button,
                        pot=pot,
                        total_contrib=tuple(total_contrib),
                        street_contrib=tuple(street_contrib),
                        current_to_call=current_to_call,
                        bet_level=bet_level,
                        acted_this_street=tuple(sorted(acted)),
                        last_aggressor=last_aggressor,
                        history_by_street=tuple(history_by_street),
                    )
                )
            return (
                PublicTreeState(
                    street=state.street,
                    current_player=opponent,
                    button=state.button,
                    pot=pot,
                    total_contrib=tuple(total_contrib),
                    street_contrib=tuple(street_contrib),
                    current_to_call=current_to_call,
                    bet_level=bet_level,
                    acted_this_street=tuple(sorted(acted)),
                    last_aggressor=last_aggressor,
                    history_by_street=tuple(history_by_street),
                ),
                None,
            )

        bet_size = config.small_bet if state.street in {Street.PREFLOP, Street.FLOP} else config.big_bet
        total_contrib[player] += to_call + bet_size
        street_contrib[player] += to_call + bet_size
        pot += to_call + bet_size
        current_to_call += bet_size
        bet_level += 1
        history_by_street[street_index] += code
        return (
            PublicTreeState(
                street=state.street,
                current_player=opponent,
                button=state.button,
                pot=pot,
                total_contrib=tuple(total_contrib),
                street_contrib=tuple(street_contrib),
                current_to_call=current_to_call,
                bet_level=bet_level,
                acted_this_street=(player,),
                last_aggressor=player,
                history_by_street=tuple(history_by_street),
            ),
            None,
        )

    def _advance_public_street(self, state: PublicTreeState) -> tuple[PublicTreeState, str | None]:
        if state.street == Street.RIVER:
            terminal = PublicTreeState(
                street=Street.RIVER,
                current_player=state.current_player,
                button=state.button,
                pot=state.pot,
                total_contrib=state.total_contrib,
                street_contrib=(0, 0),
                current_to_call=0,
                bet_level=0,
                acted_this_street=(),
                last_aggressor=None,
                history_by_street=state.history_by_street,
                terminal_type="showdown",
                folded_player=None,
            )
            return terminal, None
        next_street = {
            Street.PREFLOP: Street.FLOP,
            Street.FLOP: Street.TURN,
            Street.TURN: Street.RIVER,
        }[state.street]
        next_state = PublicTreeState(
            street=next_street,
            current_player=1 - state.button,
            button=state.button,
            pot=state.pot,
            total_contrib=state.total_contrib,
            street_contrib=(0, 0),
            current_to_call=0,
            bet_level=0,
            acted_this_street=(),
            last_aggressor=None,
            history_by_street=state.history_by_street,
        )
        return next_state, state.street.value

    def _sample_abstraction_tables(
        self,
        config: AbstractHULHEConfig,
    ) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]], dict[str, float]]:
        samples = max(config.abstraction_samples, config.river_payoff_samples)
        initial_counts: dict[str, int] = {}
        transitions_raw: dict[str, dict[str, dict[str, int]]] = {
            Street.PREFLOP.value: {},
            Street.FLOP.value: {},
            Street.TURN.value: {},
        }
        payoff_sum: dict[str, float] = {}
        payoff_count: dict[str, int] = {}

        for index in range(samples):
            deck = shuffled_deck(stable_seed((config.seed, "abstract-sample", index)))
            p0_hole = (deck[0], deck[2])
            p1_hole = (deck[1], deck[3])
            flop = tuple(deck[4:7])
            turn_board = tuple(deck[4:8])
            river_board = tuple(deck[4:9])

            pre_pair = pair_key(
                self.bucketer.bucket(Street.PREFLOP, p0_hole, ()),
                self.bucketer.bucket(Street.PREFLOP, p1_hole, ()),
            )
            flop_pair = pair_key(
                self.bucketer.bucket(Street.FLOP, p0_hole, flop),
                self.bucketer.bucket(Street.FLOP, p1_hole, flop),
            )
            turn_pair = pair_key(
                self.bucketer.bucket(Street.TURN, p0_hole, turn_board),
                self.bucketer.bucket(Street.TURN, p1_hole, turn_board),
            )
            river_pair = pair_key(
                self.bucketer.bucket(Street.RIVER, p0_hole, river_board),
                self.bucketer.bucket(Street.RIVER, p1_hole, river_board),
            )

            initial_counts[pre_pair] = initial_counts.get(pre_pair, 0) + 1
            self._count_transition(transitions_raw[Street.PREFLOP.value], pre_pair, flop_pair)
            self._count_transition(transitions_raw[Street.FLOP.value], flop_pair, turn_pair)
            self._count_transition(transitions_raw[Street.TURN.value], turn_pair, river_pair)

            result = compare_hands(tuple(p0_hole + river_board), tuple(p1_hole + river_board))
            share = 1.0 if result > 0 else 0.5 if result == 0 else 0.0
            payoff_sum[river_pair] = payoff_sum.get(river_pair, 0.0) + share
            payoff_count[river_pair] = payoff_count.get(river_pair, 0) + 1

        initial_distribution = self._normalize_counts(initial_counts)
        transitions = {
            street: {
                source: self._normalize_counts(destinations)
                for source, destinations in sources.items()
            }
            for street, sources in transitions_raw.items()
        }
        payoff = {
            pair: payoff_sum[pair] / payoff_count[pair]
            for pair in payoff_sum
        }
        payoff = self._symmetrize_showdown(payoff)
        return initial_distribution, transitions, payoff

    @staticmethod
    def _count_transition(mapping: dict[str, dict[str, int]], source: str, destination: str) -> None:
        bucket = mapping.setdefault(source, {})
        bucket[destination] = bucket.get(destination, 0) + 1

    @staticmethod
    def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in counts.items()}

    @staticmethod
    def _symmetrize_showdown(payoff: dict[str, float]) -> dict[str, float]:
        adjusted = dict(payoff)
        for pair, value in list(payoff.items()):
            left, right = pair.split("-")
            opposite = f"{right}-{left}"
            if opposite in payoff:
                corrected = (value + (1.0 - payoff[opposite])) / 2.0
                adjusted[pair] = corrected
                adjusted[opposite] = 1.0 - corrected
            else:
                adjusted[opposite] = 1.0 - value
        return adjusted
