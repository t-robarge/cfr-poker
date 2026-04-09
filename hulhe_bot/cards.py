from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import random
from typing import Iterable

from .models import Street

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}
SUITS = "cdhs"
DECK = tuple(f"{rank}{suit}" for rank in RANK_ORDER for suit in SUITS)
DESCENDING_RANKS = "AKQJT98765432"


def stable_seed(parts: Iterable[object]) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def shuffled_deck(seed: int) -> list[str]:
    rng = random.Random(seed)
    cards = list(DECK)
    rng.shuffle(cards)
    return cards


def canonicalize_cards(cards: Iterable[str]) -> tuple[str, ...]:
    mapping: dict[str, str] = {}
    next_index = 0
    canonical: list[str] = []
    for card in cards:
        rank, suit = card[0], card[1]
        if suit not in mapping:
            mapping[suit] = SUITS[next_index]
            next_index += 1
        canonical.append(f"{rank}{mapping[suit]}")
    return tuple(canonical)


def canonical_state_key(street: Street, hole_cards: tuple[str, str], board: tuple[str, ...]) -> str:
    ordered_hole = tuple(sorted(hole_cards, key=lambda card: (RANK_VALUE[card[0]], card[1]), reverse=True))
    canonical = canonicalize_cards(ordered_hole + board)
    return f"{street.value}:{','.join(canonical)}"


def canonicalize_hole_board(
    hole_cards: tuple[str, str],
    board: tuple[str, ...],
) -> tuple[tuple[str, str], tuple[str, ...]]:
    ordered_hole = tuple(sorted(hole_cards, key=lambda card: (RANK_VALUE[card[0]], card[1]), reverse=True))
    canonical = canonicalize_cards(ordered_hole + board)
    return (canonical[:2], canonical[2:])


@lru_cache(maxsize=500_000)
def evaluate_5card(cards: tuple[str, ...]) -> tuple[int, ...]:
    ranks = sorted((RANK_VALUE[card[0]] for card in cards), reverse=True)
    suits = [card[1] for card in cards]
    counts = Counter(ranks)
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    distinct_ranks = sorted(counts.keys(), reverse=True)

    is_flush = len(set(suits)) == 1
    wheel = [14, 5, 4, 3, 2]
    straight_high = 0
    if len(distinct_ranks) == 5:
        if distinct_ranks == wheel:
            straight_high = 5
        elif distinct_ranks[0] - distinct_ranks[-1] == 4:
            straight_high = distinct_ranks[0]
    if is_flush and straight_high:
        return (8, straight_high)
    if ordered[0][1] == 4:
        quad = ordered[0][0]
        kicker = ordered[1][0]
        return (7, quad, kicker)
    if ordered[0][1] == 3 and ordered[1][1] == 2:
        return (6, ordered[0][0], ordered[1][0])
    if is_flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if ordered[0][1] == 3:
        trips = ordered[0][0]
        kickers = sorted((rank for rank, count in ordered[1:] for _ in range(count)), reverse=True)
        return (3, trips, *kickers)
    if ordered[0][1] == 2 and ordered[1][1] == 2:
        high_pair = max(ordered[0][0], ordered[1][0])
        low_pair = min(ordered[0][0], ordered[1][0])
        kicker = ordered[2][0]
        return (2, high_pair, low_pair, kicker)
    if ordered[0][1] == 2:
        pair = ordered[0][0]
        kickers = sorted((rank for rank, count in ordered[1:] for _ in range(count)), reverse=True)
        return (1, pair, *kickers)
    return (0, *ranks)


@lru_cache(maxsize=500_000)
def evaluate_7card(cards: tuple[str, ...]) -> tuple[int, ...]:
    if len(cards) != 7:
        raise ValueError("evaluate_7card expects exactly seven cards.")
    return evaluate_best(cards)


@lru_cache(maxsize=500_000)
def evaluate_best(cards: tuple[str, ...]) -> tuple[int, ...]:
    if len(cards) < 5:
        raise ValueError("Need at least five cards to evaluate a hand.")
    best = None
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5card(combo)
        if best is None or score > best:
            best = score
    if best is None:
        raise ValueError("Need at least five cards to evaluate a hand.")
    return best


def compare_hands(cards_a: tuple[str, ...], cards_b: tuple[str, ...]) -> int:
    score_a = evaluate_best(cards_a)
    score_b = evaluate_best(cards_b)
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


@lru_cache(maxsize=100_000)
def exact_showdown_share(hole_cards: tuple[str, str], board: tuple[str, ...]) -> float:
    canonical_hole, canonical_board = canonicalize_hole_board(hole_cards, board)
    known = set(canonical_hole) | set(canonical_board)
    remaining = [card for card in DECK if card not in known]
    wins = 0.0
    total = 0
    hero_cards = tuple(canonical_hole + canonical_board)
    hero_score = evaluate_7card(hero_cards)
    for opponent_hole in itertools.combinations(remaining, 2):
        villain_cards = tuple(opponent_hole + canonical_board)
        villain_score = evaluate_7card(villain_cards)
        result = 1 if hero_score > villain_score else -1 if villain_score > hero_score else 0
        total += 1
        if result > 0:
            wins += 1.0
        elif result == 0:
            wins += 0.5
    if total == 0:
        return 0.5
    return wins / total


def monte_carlo_showdown_share(
    hole_cards: tuple[str, str],
    board: tuple[str, ...],
    samples: int,
    seed: int,
) -> float:
    canonical_hole, canonical_board = canonicalize_hole_board(hole_cards, board)
    known = set(canonical_hole) | set(canonical_board)
    remaining = [card for card in DECK if card not in known]
    board_to_draw = 5 - len(canonical_board)
    wins = 0.0
    rng = random.Random(seed)
    for _ in range(max(1, samples)):
        draw = rng.sample(remaining, board_to_draw + 2)
        future_board = tuple(draw[:board_to_draw])
        opponent_hole = tuple(draw[board_to_draw:])
        final_board = tuple(canonical_board + future_board)
        result = compare_hands(tuple(canonical_hole + final_board), tuple(opponent_hole + final_board))
        if result > 0:
            wins += 1.0
        elif result == 0:
            wins += 0.5
    return wins / max(1, samples)


def bucket_from_percentile(percentile: float, bucket_count: int) -> int:
    clamped = min(max(percentile, 0.0), 0.999999)
    return min(bucket_count - 1, int(clamped * bucket_count))


def preflop_class_index(hole_cards: tuple[str, str]) -> int:
    first, second = hole_cards
    r1, r2 = first[0], second[0]
    s1, s2 = first[1], second[1]
    i1 = DESCENDING_RANKS.index(r1)
    i2 = DESCENDING_RANKS.index(r2)
    if i1 == i2:
        row = i1
        col = i2
    elif s1 == s2:
        row = min(i1, i2)
        col = max(i1, i2)
    else:
        row = max(i1, i2)
        col = min(i1, i2)
    return row * 13 + col


def made_hand_class(hole_cards: tuple[str, str], board: tuple[str, ...]) -> int:
    score = evaluate_best(tuple(hole_cards + board))
    rank_class = score[0]
    if rank_class == 0:
        return 0
    if rank_class == 1:
        pair_rank = score[1]
        board_top = max((RANK_VALUE[card[0]] for card in board), default=0)
        if pair_rank >= board_top:
            return 2
        return 1
    if rank_class in {2, 3}:
        return 3
    return 4


def has_flush_draw(hole_cards: tuple[str, str], board: tuple[str, ...]) -> bool:
    cards = hole_cards + board
    suit_counts = Counter(card[1] for card in cards)
    max_suit = max(suit_counts.values(), default=0)
    if len(board) >= 5:
        return False
    if max_suit >= 5:
        return False
    return max_suit >= 4


def has_straight_draw(hole_cards: tuple[str, str], board: tuple[str, ...]) -> bool:
    if len(board) >= 5:
        return False
    rank_class = evaluate_best(tuple(hole_cards + board))[0]
    if rank_class >= 4:
        return False
    ranks = {RANK_VALUE[card[0]] for card in hole_cards + board}
    if 14 in ranks:
        ranks.add(1)
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if len(window & ranks) >= 4:
            return True
    return False


def draw_class(hole_cards: tuple[str, str], board: tuple[str, ...]) -> int:
    flush_draw = has_flush_draw(hole_cards, board)
    straight_draw = has_straight_draw(hole_cards, board)
    return int(flush_draw) + int(straight_draw)
