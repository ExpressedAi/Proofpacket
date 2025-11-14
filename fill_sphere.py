#!/usr/bin/env python3
"""
FILL THE SPHERE - Quiet autonomous exploration
Just keeps building pathways until you stop it.
"""

import numpy as np
import time
import random
import sys
from generative_learning_engine import GenerativeLearningEngine, ReasoningRequest


def main():
    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="Fill the sphere with pathways")
    parser.add_argument('--sessions', type=int, default=None,
                       help='Max sessions (default: infinite, use Ctrl+C to stop)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    args = parser.parse_args()

    # Initialize (suppress verbose output)
    if args.quiet:
        import os
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

    engine = GenerativeLearningEngine()
    all_concepts = [c.name for c in engine.semantic_sphere.concepts]

    if args.quiet:
        sys.stdout = old_stdout
        devnull.close()

    print(f"\n🌐 FILLING THE SPHERE")
    print(f"   Concepts: {len(all_concepts)}")
    print(f"   Strategy: frontier exploration")
    print(f"   Max sessions: {args.sessions if args.sessions else '∞ (Ctrl+C to stop)'}")
    print()

    start_time = time.time()
    session = 0

    try:
        while True:
            if args.sessions and session >= args.sessions:
                break

            session += 1

            # Pick concepts
            memory = engine.pathway_memory
            hubs = {}
            for (from_c, to_c), stats in memory.transitions.items():
                hubs[from_c] = hubs.get(from_c, 0) + 1

            if hubs and random.random() < 0.7:
                from_c = random.choices(list(hubs.keys()), weights=list(hubs.values()))[0]
            else:
                from_c = random.choice(all_concepts)

            explored = {to_c for (f, to_c) in memory.transitions.keys() if f == from_c}
            unexplored = [c for c in all_concepts if c != from_c and c not in explored]
            to_c = random.choice(unexplored) if unexplored else random.choice(
                [c for c in all_concepts if c != from_c]
            )

            # Run reasoning (suppress output)
            old_stdout = sys.stdout
            if args.quiet:
                sys.stdout = open('/dev/null', 'w')

            request = ReasoningRequest(
                problem=f"Transition from {from_c} to {to_c}",
                context={'session': session},
                initial_concepts=[from_c],
                target_concepts=[to_c],
                max_time=3.0,
                session_id=session
            )

            result = engine.reason(request)

            if args.quiet:
                sys.stdout.close()
            sys.stdout = old_stdout

            # Quick report every 10 sessions
            if session % 10 == 0:
                num_pathways = len(memory.transitions)
                elapsed = time.time() - start_time
                rate = session / elapsed
                snaps = len(result.snaps)

                # Find average pathway strength
                if memory.transitions:
                    avg_strength = sum(s.strength for s in memory.transitions.values()) / len(memory.transitions)
                else:
                    avg_strength = 0

                print(f"Session {session:4d} | "
                      f"Pathways: {num_pathways:4d} | "
                      f"Avg strength: {avg_strength:.3f} | "
                      f"Rate: {rate:.1f}/s")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Stopped by user")

    # Final stats
    elapsed = time.time() - start_time
    num_pathways = len(memory.transitions)
    num_attractors = len(memory.attractors)

    print(f"\n")
    print(f"{'='*60}")
    print(f"SPHERE FILLING COMPLETE")
    print(f"{'='*60}")
    print(f"Sessions:    {session}")
    print(f"Time:        {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Pathways:    {num_pathways}")
    print(f"Attractors:  {num_attractors}")
    print(f"Coverage:    {100*num_pathways/(len(all_concepts)**2):.2f}% of possible transitions")
    print()

    # Top 10 strongest pathways
    strong = sorted(
        [(f, t, s.strength, s.usage_count) for (f, t), s in memory.transitions.items()],
        key=lambda x: x[2],
        reverse=True
    )[:10]

    if strong:
        print("STRONGEST PATHWAYS:")
        for i, (f, t, strength, uses) in enumerate(strong, 1):
            print(f"  {i:2d}. {f:20s} → {t:20s}  {strength:.3f} ({uses} uses)")

    print()
    engine.save_infrastructure("learning_engine_state.json")
    print(f"✓ State saved to learning_engine_state.json")


if __name__ == '__main__':
    main()
