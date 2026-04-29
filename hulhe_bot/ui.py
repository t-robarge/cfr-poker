from __future__ import annotations

from dataclasses import dataclass, field
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import threading
import urllib.parse
import uuid

from .baselines import PolicyAgent
from .config import AbstractHULHEConfig
from .engine import GameState, LimitHoldemGame
from .evaluator import Evaluator
from .models import Action, Observation, PolicyArtifact
from .policy import PolicyRuntime


SUIT_SYMBOLS = {
    "s": "♠",
    "h": "♥",
    "d": "♦",
    "c": "♣",
}

RED_SUITS = {"h", "d"}


def _sample_action(distribution: dict[Action, float], rng: random.Random) -> Action:
    threshold = rng.random()
    cumulative = 0.0
    for action, probability in distribution.items():
        cumulative += probability
        if threshold <= cumulative:
            return action
    return next(iter(distribution))


def _format_distribution(distribution: dict[Action, float]) -> str:
    return ", ".join(
        f"{action.value}:{probability:.2f}" for action, probability in distribution.items()
    )


def _render_card(card: str, *, hidden: bool = False) -> str:
    if hidden or len(card) < 2 or card == "??":
        return (
            '<div class="card hidden">'
            '<div class="card-back-pattern"></div>'
            "</div>"
        )
    rank = html.escape(card[0])
    suit_key = card[1].lower()
    suit = html.escape(SUIT_SYMBOLS.get(suit_key, suit_key))
    color_class = "red" if suit_key in RED_SUITS else "black"
    return (
        f'<div class="card {color_class}">'
        f'<div class="corner top"><span class="rank">{rank}</span><span class="suit">{suit}</span></div>'
        f'<div class="center-suit">{suit}</div>'
        f'<div class="corner bottom"><span class="rank">{rank}</span><span class="suit">{suit}</span></div>'
        "</div>"
    )


def _render_card_row(cards: tuple[str, ...], *, hidden: bool = False, empty_text: str = "") -> str:
    if not cards:
        return f'<div class="card-row empty">{html.escape(empty_text)}</div>' if empty_text else '<div class="card-row"></div>'
    return '<div class="card-row">' + "".join(_render_card(card, hidden=hidden) for card in cards) + "</div>"


def _discover_options(pattern: str) -> list[str]:
    root = Path.cwd()
    values: list[str] = []
    for path in sorted(root.glob(pattern)):
        try:
            values.append(str(path.relative_to(root)))
        except ValueError:
            values.append(str(path))
    return values


def _discover_policy_options() -> list[str]:
    root = Path.cwd()
    values: list[str] = []
    for path in sorted(root.glob("artifacts/*.json")):
        try:
            PolicyArtifact.load(path)
        except Exception:
            continue
        try:
            values.append(str(path.relative_to(root)))
        except ValueError:
            values.append(str(path))
    return values


@dataclass(slots=True)
class DecisionLogEntry:
    actor: str
    street: str
    action: str
    pot: int
    history_id: str
    note: str = ""


@dataclass(slots=True)
class ModelPlaySession:
    session_id: str
    config_path: str
    policy_path: str
    human_seat: int
    config: AbstractHULHEConfig
    artifact: PolicyArtifact
    evaluator: Evaluator
    runtime: PolicyRuntime
    game: LimitHoldemGame
    rng: random.Random
    hand_index: int = 0
    human_total_sb: float = 0.0
    model_total_sb: float = 0.0
    state: GameState | None = None
    hand_log: list[DecisionLogEntry] = field(default_factory=list)
    last_result_text: str = ""
    _result_recorded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start_new_hand(self) -> None:
        self.hand_index += 1
        seed = self.config.seed * 100_003 + self.hand_index
        button = self.hand_index % 2
        self.state = self.game.new_hand(seed=seed, button=button)
        self.hand_log = []
        self.last_result_text = ""
        self._result_recorded = False
        self._advance_until_human()

    def apply_human_action(self, action_name: str) -> None:
        if self.state is None or self.state.terminal:
            return
        if self.state.current_player != self.human_seat:
            return
        action = Action(action_name)
        if action not in self.game.legal_actions(self.state):
            return
        before = self.state
        self.hand_log.append(
            DecisionLogEntry(
                actor="you",
                street=before.street.value,
                action=action.value,
                pot=before.pot,
                history_id=before.history_id(),
            )
        )
        self.state = self.game.apply_action(self.state, action)
        self._advance_until_human()

    def current_observation(self) -> Observation | None:
        if self.state is None or self.state.terminal:
            return None
        if self.state.current_player != self.human_seat:
            return None
        return self.evaluator.make_observation(self.state)

    def available_actions(self) -> tuple[Action, ...]:
        if self.state is None or self.state.terminal:
            return ()
        if self.state.current_player != self.human_seat:
            return ()
        return self.game.legal_actions(self.state)

    def status(self) -> dict[str, object]:
        state = self.state
        if state is None:
            raise ValueError("Session has not been initialized")
        return {
            "hand_index": self.hand_index,
            "street": state.street.value,
            "board_cards": state.board,
            "pot": state.pot,
            "history_id": state.history_id(),
            "your_cards": state.hole_cards[self.human_seat],
            "opponent_cards": state.hole_cards[1 - self.human_seat] if state.terminal else ("??", "??"),
            "button": state.button,
            "current_player": state.current_player,
            "terminal": state.terminal,
            "human_total_sb": self.human_total_sb,
            "model_total_sb": self.model_total_sb,
            "last_result_text": self.last_result_text,
            "mode": getattr(self.config, "subgame_mode", "heuristic"),
            "algorithm": self.artifact.algorithm,
            "has_nn": bool(self.artifact.nn_weights),
            "has_residual": bool(self.artifact.residual_table) or bool(self.artifact.nn_weights),
        }

    def _advance_until_human(self) -> None:
        while self.state is not None and not self.state.terminal and self.state.current_player != self.human_seat:
            observation = self.evaluator.make_observation(self.state)
            distribution = self.runtime.distribution(observation)
            action = _sample_action(distribution, self.rng)
            self.hand_log.append(
                DecisionLogEntry(
                    actor="model",
                    street=observation.street.value,
                    action=action.value,
                    pot=observation.pot,
                    history_id=observation.history_id,
                    note=_format_distribution(distribution),
                )
            )
            self.state = self.game.apply_action(self.state, action)
        self._record_terminal_result()

    def _record_terminal_result(self) -> None:
        if self.state is None or not self.state.terminal or self._result_recorded:
            return
        human_payoff = self.state.payoffs[self.human_seat]
        model_payoff = self.state.payoffs[1 - self.human_seat]
        self.human_total_sb += human_payoff
        self.model_total_sb += model_payoff
        winner = "you" if human_payoff > model_payoff else "model"
        if human_payoff == model_payoff:
            winner = "tie"
        self.last_result_text = (
            f"Hand finished: {winner}. Your payoff {human_payoff:+.2f} sb, "
            f"model payoff {model_payoff:+.2f} sb."
        )
        self._result_recorded = True


class UISessionManager:
    def __init__(self):
        self._sessions: dict[str, ModelPlaySession] = {}
        self._lock = threading.Lock()

    def create(self, config_path: str, policy_path: str, human_seat: int) -> ModelPlaySession:
        config = AbstractHULHEConfig.load(config_path)
        artifact = PolicyArtifact.load(policy_path)
        evaluator = Evaluator(config)
        runtime = PolicyRuntime(artifact, config, evaluator.bucketer)
        session = ModelPlaySession(
            session_id=str(uuid.uuid4()),
            config_path=config_path,
            policy_path=policy_path,
            human_seat=human_seat,
            config=config,
            artifact=artifact,
            evaluator=evaluator,
            runtime=runtime,
            game=LimitHoldemGame(config),
            rng=random.Random(config.seed + human_seat * 13),
        )
        session.start_new_hand()
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ModelPlaySession | None:
        with self._lock:
            return self._sessions.get(session_id)


def _render_root_page(
    *,
    session: ModelPlaySession | None,
    config_options: list[str],
    policy_options: list[str],
    default_config: str | None,
    default_policy: str | None,
) -> str:
    def option_list(values: list[str], selected: str | None) -> str:
        items = []
        for value in values:
            is_selected = " selected" if value == selected else ""
            items.append(f'<option value="{html.escape(value)}"{is_selected}>{html.escape(value)}</option>')
        return "\n".join(items)

    launcher = f"""
    <section class=\"panel\">
      <h2>Start or reload a model session</h2>
      <form method=\"POST\" action=\"/new-session\">
        <label>Config</label>
        <select name=\"config\">{option_list(config_options, default_config)}</select>
        <label>Policy artifact</label>
        <select name=\"policy\">{option_list(policy_options, default_policy)}</select>
        <label>Your seat</label>
        <select name=\"human_seat\">
          <option value=\"0\">Seat 0</option>
          <option value=\"1\">Seat 1</option>
        </select>
        <button type=\"submit\">Start session</button>
      </form>
    </section>
    """

    if session is None:
        body = "<section class=\"panel\"><p>No active session yet.</p></section>"
    else:
        status = session.status()
        observation = session.current_observation()
        board_html = _render_card_row(status["board_cards"], empty_text="Board will appear after the flop")
        hero_cards_html = _render_card_row(status["your_cards"])
        villain_cards_html = _render_card_row(status["opponent_cards"], hidden=not bool(status["terminal"]))
        actions_html = ""
        for action in session.available_actions():
            actions_html += (
                f'<form method="POST" action="/action?session={html.escape(session.session_id)}" class="inline">'
                f'<input type="hidden" name="action" value="{html.escape(action.value)}">'
                f'<button type="submit">{html.escape(action.value.title())}</button>'
                "</form>"
            )
        if not actions_html:
            actions_html = "<p>Waiting for model or hand complete.</p>"

        observation_html = ""
        if observation is not None:
            observation_html = f"""
            <ul>
              <li>Bucket id: {observation.bucket_id}</li>
              <li>Bucket percentile: {observation.bucket_percentile:.3f}</li>
              <li>To call: {observation.to_call}</li>
              <li>Legal actions: {", ".join(action.value for action in observation.legal_actions)}</li>
            </ul>
            """

        log_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(entry.actor)}</td>"
            f"<td>{html.escape(entry.street)}</td>"
            f"<td>{html.escape(entry.action)}</td>"
            f"<td>{entry.pot}</td>"
            f"<td>{html.escape(entry.history_id)}</td>"
            f"<td>{html.escape(entry.note)}</td>"
            "</tr>"
            for entry in session.hand_log
        ) or "<tr><td colspan=\"6\">No actions yet.</td></tr>"

        body = f"""
        <section class=\"grid\">
          <div class=\"panel\">
            <h2>Current hand</h2>
                        <div class="table-area">
                            <div class="seat-block">
                                <div class="seat-label">Opponent</div>
                                {villain_cards_html}
                            </div>
                            <div class="board-block">
                                <div class="seat-label">Board</div>
                                {board_html}
                            </div>
                            <div class="seat-block">
                                <div class="seat-label">You</div>
                                {hero_cards_html}
                            </div>
                        </div>
            <ul>
              <li>Hand #: {status['hand_index']}</li>
              <li>Street: {html.escape(str(status['street']))}</li>
              <li>Pot: {status['pot']}</li>
              <li>History id: {html.escape(str(status['history_id']))}</li>
              <li>Button seat: {status['button']}</li>
            </ul>
            <p><strong>{html.escape(str(status['last_result_text']))}</strong></p>
            <form method=\"POST\" action=\"/new-hand?session={html.escape(session.session_id)}\">
              <button type=\"submit\">Deal next hand</button>
            </form>
          </div>
          <div class=\"panel\">
            <h2>Loaded model</h2>
            <ul>
              <li>Config: {html.escape(session.config_path)}</li>
              <li>Policy: {html.escape(session.policy_path)}</li>
              <li>Blueprint algorithm: {html.escape(str(status['algorithm']))}</li>
              <li>Residual loaded: {status['has_residual']}</li>
              <li>Neural residual: {status['has_nn']}</li>
              <li>Subgame mode: {html.escape(str(status['mode']))}</li>
            </ul>
            <h3>Match totals</h3>
            <ul>
              <li>Your total: {status['human_total_sb']:+.2f} sb</li>
              <li>Model total: {status['model_total_sb']:+.2f} sb</li>
            </ul>
          </div>
          <div class=\"panel\">
            <h2>Your decision</h2>
            {observation_html}
            <div class=\"actions\">{actions_html}</div>
          </div>
        </section>
        <section class=\"panel\">
          <h2>Action log</h2>
          <table>
            <thead>
              <tr><th>Actor</th><th>Street</th><th>Action</th><th>Pot</th><th>History</th><th>Model distribution</th></tr>
            </thead>
            <tbody>{log_rows}</tbody>
          </table>
        </section>
        """

    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\">
        <title>HULHE Model UI</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }}
          h1, h2, h3 {{ margin-top: 0; }}
          .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
          .panel {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
                    .table-area {{ background: radial-gradient(circle at top, #14532d, #052e16); border: 1px solid #166534; border-radius: 18px; padding: 18px; margin-bottom: 16px; }}
                    .seat-block, .board-block {{ margin-bottom: 14px; }}
                    .seat-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #cbd5e1; margin-bottom: 8px; }}
                    .card-row {{ display: flex; flex-wrap: wrap; gap: 10px; min-height: 88px; align-items: center; }}
                    .card-row.empty {{ color: #cbd5e1; font-size: 14px; }}
                    .card {{ position: relative; width: 58px; height: 82px; border-radius: 10px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); border: 1px solid #cbd5e1; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.32); }}
                    .card.red {{ color: #dc2626; }}
                    .card.black {{ color: #0f172a; }}
                    .card.hidden {{ background: linear-gradient(145deg, #1d4ed8, #1e3a8a); border: 1px solid #93c5fd; overflow: hidden; }}
                    .card-back-pattern {{ position: absolute; inset: 6px; border-radius: 8px; background:
                        repeating-linear-gradient(45deg, rgba(255,255,255,0.18), rgba(255,255,255,0.18) 4px, transparent 4px, transparent 8px),
                        repeating-linear-gradient(-45deg, rgba(255,255,255,0.16), rgba(255,255,255,0.16) 4px, transparent 4px, transparent 8px); }}
                    .corner {{ position: absolute; display: flex; flex-direction: column; align-items: center; line-height: 1; font-weight: 700; font-size: 13px; }}
                    .corner.top {{ top: 5px; left: 6px; }}
                    .corner.bottom {{ right: 6px; bottom: 5px; transform: rotate(180deg); }}
                    .center-suit {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 700; }}
          label {{ display: block; margin: 8px 0 4px; }}
          select, button {{ font-size: 14px; padding: 8px 10px; border-radius: 8px; border: 1px solid #475569; background: #1e293b; color: #e2e8f0; }}
          button {{ cursor: pointer; }}
          .actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
          .inline {{ display: inline-block; }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ border-bottom: 1px solid #334155; padding: 8px; text-align: left; font-size: 13px; }}
          code {{ color: #93c5fd; }}
        </style>
      </head>
      <body>
        <h1>HULHE model playground</h1>
        <p>Play against the loaded artifact. Blueprint, residual policy, and live subgame solving all run through the normal runtime path.</p>
        {launcher}
        {body}
      </body>
    </html>
    """


def launch_ui(
    *,
    config_path: str | None,
    policy_path: str | None,
    host: str,
    port: int,
    human_seat: int,
) -> None:
    manager = UISessionManager()
    config_options = _discover_options("configs/*.json")
    policy_options = _discover_policy_options()
    default_config = config_path or (config_options[0] if config_options else None)
    default_policy = policy_path or (policy_options[0] if policy_options else None)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            params = urllib.parse.parse_qs(parsed.query)
            session_id = params.get("session", [""])[0]
            session = manager.get(session_id) if session_id else None
            page = _render_root_page(
                session=session,
                config_options=config_options,
                policy_options=policy_options,
                default_config=default_config,
                default_policy=default_policy,
            )
            self._send_html(page)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            form = self._read_form()
            if parsed.path == "/new-session":
                config_value = form.get("config", [default_config or ""])[0]
                policy_value = form.get("policy", [default_policy or ""])[0]
                human_value = int(form.get("human_seat", [str(human_seat)])[0])
                session = manager.create(config_value, policy_value, human_value)
                self._redirect(f"/?session={session.session_id}")
                return

            params = urllib.parse.parse_qs(parsed.query)
            session_id = params.get("session", [""])[0]
            session = manager.get(session_id)
            if session is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown session")
                return

            with session.lock:
                if parsed.path == "/action":
                    action_value = form.get("action", [""])[0]
                    session.apply_human_action(action_value)
                elif parsed.path == "/new-hand":
                    session.start_new_hand()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            self._redirect(f"/?session={session.session_id}")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _read_form(self) -> dict[str, list[str]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8") if content_length else ""
            return urllib.parse.parse_qs(raw)

        def _redirect(self, target: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", target)
            self.end_headers()

        def _send_html(self, payload: str) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[UI] HULHE model UI running at http://{host}:{port}")
    if default_config and default_policy:
        print(f"[UI] Default config: {default_config}")
        print(f"[UI] Default policy: {default_policy}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()