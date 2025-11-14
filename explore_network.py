#!/usr/bin/env python3
"""Quick network viewer - just show the structure"""

import json
from collections import defaultdict

# Load state
with open('learning_engine_state.json', 'r') as f:
    data = json.load(f)

transitions = data['transitions']

print(f"\n{'='*80}")
print(f"PATHWAY NETWORK SNAPSHOT")
print(f"{'='*80}\n")

# Count hubs and attractors
outgoing = defaultdict(int)
incoming = defaultdict(int)
for key in transitions.keys():
    from_c, to_c = key.split('→')
    outgoing[from_c] += 1
    incoming[to_c] += 1

print(f"Total pathways: {len(transitions)}")
print(f"Unique sources: {len(outgoing)}")
print(f"Unique targets: {len(incoming)}")
print()

# Top hubs
print("TOP 10 HUBS (most outgoing pathways):")
for i, (concept, count) in enumerate(sorted(outgoing.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"  {i:2d}. {concept:30s} → {count:3d} pathways")

print()

# Top attractors
print("TOP 10 ATTRACTORS (most incoming pathways):")
for i, (concept, count) in enumerate(sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"  {i:2d}. {concept:30s} ← {count:3d} pathways")

print()

# Pathway strength distribution (calculate like pathway_memory.py)
def calc_strength(stats):
    usage_factor = min(1.0, stats['usage_count'] / 50.0)
    success_factor = stats['success_rate']
    speed_factor = 1.0 / (1.0 + stats['average_convergence_time'])
    validation_factor = stats['rg_persistence_score']
    return 0.4 * usage_factor + 0.3 * success_factor + 0.2 * speed_factor + 0.1 * validation_factor

strengths = []
for key, stats in transitions.items():
    strengths.append(calc_strength(stats))

strengths.sort(reverse=True)

print("PATHWAY STRENGTH DISTRIBUTION:")
print(f"  Strongest:  {strengths[0]:.3f}")
print(f"  Top 10 avg: {sum(strengths[:10])/10:.3f}")
print(f"  Top 50 avg: {sum(strengths[:50])/50:.3f}")
print(f"  Overall:    {sum(strengths)/len(strengths):.3f}")
print()

# Strong pathways
strong = [(k.split('→'), calc_strength(v), v['usage_count']) for k, v in transitions.items()]
strong.sort(key=lambda x: x[1], reverse=True)

print("TOP 20 STRONGEST PATHWAYS:")
for i, ((from_c, to_c), strength, uses) in enumerate(strong[:20], 1):
    bar_len = int(strength * 50)
    bar = '█' * bar_len
    print(f"  {i:2d}. {from_c:20s} → {to_c:20s} {bar} {strength:.3f} ({uses}×)")

print()
