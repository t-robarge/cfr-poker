from __future__ import annotations

import random

from .models import Action, Observation
from .policy import PolicyRuntime


class Agent:
    name = "agent"

    def select_action(self, observation: Observation, rng: random.Random) -> Action:
        raise NotImplementedError


class PolicyAgent(Agent):
    def __init__(self, runtime: PolicyRuntime, name: str = "policy"):
        self.runtime = runtime
        self.name = name

    def select_action(self, observation: Observation, rng: random.Random) -> Action:
        return self.runtime.act(observation, rng)


class RandomAgent(Agent):
    name = "random"

    def select_action(self, observation: Observation, rng: random.Random) -> Action:
        return rng.choice(list(observation.legal_actions))


class LoosePassiveAgent(Agent):
    name = "loose_passive"

    def select_action(self, observation: Observation, rng: random.Random) -> Action:
        if Action.CHECK in observation.legal_actions and observation.to_call == 0:
            if Action.RAISE in observation.legal_actions and observation.bucket_percentile > 0.95 and rng.random() < 0.15:
                return Action.RAISE
            return Action.CHECK
        if observation.bucket_percentile < 0.18 and Action.FOLD in observation.legal_actions and rng.random() < 0.65:
            return Action.FOLD
        if Action.RAISE in observation.legal_actions and observation.bucket_percentile > 0.93 and rng.random() < 0.25:
            return Action.RAISE
        return Action.CALL if Action.CALL in observation.legal_actions else Action.CHECK


class TightAggressiveAgent(Agent):
    name = "tight_aggressive"

    def select_action(self, observation: Observation, rng: random.Random) -> Action:
        threshold = 0.72 if observation.street.value == "preflop" else 0.75
        fold_threshold = 0.28 if observation.street.value == "preflop" else 0.35
        if observation.to_call == 0:
            if Action.RAISE in observation.legal_actions and observation.bucket_percentile >= threshold:
                return Action.RAISE
            return Action.CHECK
        if Action.FOLD in observation.legal_actions and observation.bucket_percentile < fold_threshold:
            return Action.FOLD
        if Action.RAISE in observation.legal_actions and observation.bucket_percentile >= threshold:
            return Action.RAISE
        return Action.CALL if Action.CALL in observation.legal_actions else Action.CHECK

