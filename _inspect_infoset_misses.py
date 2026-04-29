from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.evaluator import Evaluator
from hulhe_bot.models import PolicyArtifact
from hulhe_bot.policy import PolicyRuntime
from hulhe_bot.baselines import PolicyAgent
import random

config = AbstractHULHEConfig.load('configs/tag_heavy_045.json')
art = PolicyArtifact.load('artifacts/tag_heavy_045_tuned.json')
ev = Evaluator(config)
rt = PolicyRuntime(art, config, ev.bucketer)
agent = PolicyAgent(rt)
miss = total = 0
samples = []
for hand_seed in range(40):
    state = ev.game.new_hand(seed=1000 + hand_seed, button=hand_seed % 2)
    while not state.terminal:
        obs = ev.make_observation(state)
        key = rt.infoset_key(obs).encode()
        found = key in art.policy_table
        total += 1
        if not found:
            miss += 1
            if len(samples) < 12:
                samples.append({
                    'acting_player': obs.acting_player,
                    'street': obs.street.value,
                    'history_id': obs.history_id,
                    'bucket_id': obs.bucket_id,
                    'legal_actions': [a.value for a in obs.legal_actions],
                })
        action = agent.select_action(obs, random.Random(1))
        state = ev.game.apply_action(state, action)

print({'missing': miss, 'total': total, 'rate': miss / total})
for sample in samples:
    print(sample)
