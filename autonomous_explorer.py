#!/usr/bin/env python3
"""
AUTONOMOUS SPHERE EXPLORER
Continuously explores concept space, discovering and strengthening pathways.
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from generative_learning_engine import (
    GenerativeLearningEngine,
    ReasoningRequest
)
from pathway_memory import PathwayMemory
import random


class AutonomousExplorer:
    """
    Explores the sphere autonomously, finding new pathways.
    """

    def __init__(self,
                 engine: GenerativeLearningEngine,
                 exploration_strategy: str = "random"):
        """
        Args:
            engine: The generative learning engine
            exploration_strategy: "random", "frontier", or "dense"
        """
        self.engine = engine
        self.strategy = exploration_strategy
        self.session_count = 0

        # Get all available concepts
        self.all_concepts = [c.name for c in engine.semantic_sphere.concepts]
        print(f"Explorer initialized with {len(self.all_concepts)} concepts")

    def pick_exploration_target(self) -> Tuple[str, str]:
        """Pick next (from, to) concept pair to explore."""

        if self.strategy == "random":
            # Pure random exploration
            from_concept = random.choice(self.all_concepts)
            to_concept = random.choice(self.all_concepts)
            while to_concept == from_concept:
                to_concept = random.choice(self.all_concepts)
            return from_concept, to_concept

        elif self.strategy == "frontier":
            # Explore from known hubs to unknown targets
            memory = self.engine.pathway_memory

            # Find concepts with outgoing pathways (hubs)
            hubs = {}
            for (from_c, to_c), stats in memory.transitions.items():
                hubs[from_c] = hubs.get(from_c, 0) + 1

            if hubs:
                # Pick a hub
                from_concept = random.choices(
                    list(hubs.keys()),
                    weights=list(hubs.values())
                )[0]
            else:
                # No hubs yet, pick random
                from_concept = random.choice(self.all_concepts)

            # Pick unexplored target
            explored_from_here = {
                to_c for (f, to_c) in memory.transitions.keys()
                if f == from_concept
            }
            unexplored = [c for c in self.all_concepts
                         if c != from_concept and c not in explored_from_here]

            if unexplored:
                to_concept = random.choice(unexplored)
            else:
                to_concept = random.choice(self.all_concepts)
                while to_concept == from_concept:
                    to_concept = random.choice(self.all_concepts)

            return from_concept, to_concept

        elif self.strategy == "dense":
            # Strengthen existing pathways (re-traverse known roads)
            memory = self.engine.pathway_memory

            if memory.transitions:
                # 70% chance to strengthen existing pathway
                if random.random() < 0.7:
                    from_c, to_c = random.choice(list(memory.transitions.keys()))
                    return from_c, to_c

            # 30% chance (or if no pathways) explore new
            from_concept = random.choice(self.all_concepts)
            to_concept = random.choice(self.all_concepts)
            while to_concept == from_concept:
                to_concept = random.choice(self.all_concepts)
            return from_concept, to_concept

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def explore_once(self) -> dict:
        """Run one exploration session."""
        self.session_count += 1

        # Pick target
        from_concept, to_concept = self.pick_exploration_target()

        # Create reasoning problem
        problem = f"Transition from {from_concept} to {to_concept}"
        context = {
            'exploration_session': self.session_count,
            'strategy': self.strategy
        }

        # Attempt transition
        request = ReasoningRequest(
            problem=problem,
            context=context,
            initial_concepts=[from_concept],
            target_concepts=[to_concept],
            max_time=5.0,  # Limit time for speed
            session_id=self.session_count
        )

        start_time = time.time()
        result = self.engine.reason(request)
        elapsed = time.time() - start_time

        # Determine if successful
        success = (result.solution_concept == to_concept or
                  to_concept in result.trajectory)
        snapped = len(result.snaps) > 0

        return {
            'session': self.session_count,
            'from': from_concept,
            'to': to_concept,
            'success': success,
            'final_concept': result.solution_concept,
            'snapped': snapped,
            'time': elapsed,
            'steps': len(result.trajectory)
        }

    def explore_continuous(self,
                          max_sessions: Optional[int] = None,
                          max_time: Optional[float] = None,
                          report_every: int = 10):
        """
        Explore continuously until stopping condition.

        Args:
            max_sessions: Stop after N sessions (None = infinite)
            max_time: Stop after N seconds (None = infinite)
            report_every: Print report every N sessions
        """
        start_time = time.time()

        stats = {
            'total': 0,
            'successes': 0,
            'snaps': 0,
            'total_time': 0.0,
            'total_steps': 0
        }

        print("\n" + "="*80)
        print("AUTONOMOUS EXPLORATION STARTED")
        print("="*80)
        print(f"Strategy: {self.strategy}")
        print(f"Max sessions: {max_sessions if max_sessions else '∞'}")
        print(f"Max time: {max_time if max_time else '∞'}s")
        print()

        try:
            while True:
                # Check stopping conditions
                if max_sessions and stats['total'] >= max_sessions:
                    break
                if max_time and (time.time() - start_time) >= max_time:
                    break

                # Explore
                result = self.explore_once()

                # Update stats
                stats['total'] += 1
                stats['total_time'] += result['time']
                stats['total_steps'] += result['steps']
                if result['success']:
                    stats['successes'] += 1
                if result['snapped']:
                    stats['snaps'] += 1

                # Report
                if stats['total'] % report_every == 0:
                    self.print_progress_report(stats, time.time() - start_time)

        except KeyboardInterrupt:
            print("\n\n⚠️  Exploration interrupted by user")

        # Final report
        print("\n" + "="*80)
        print("EXPLORATION COMPLETE")
        print("="*80)
        self.print_final_report(stats, time.time() - start_time)

        # Save state
        state_file = "learning_engine_state.json"
        self.engine.save_infrastructure(state_file)
        print(f"\n✓ State saved to: {state_file}")

    def print_progress_report(self, stats: dict, elapsed: float):
        """Print progress during exploration."""
        success_rate = 100 * stats['successes'] / stats['total'] if stats['total'] > 0 else 0
        snap_rate = 100 * stats['snaps'] / stats['total'] if stats['total'] > 0 else 0
        avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
        avg_steps = stats['total_steps'] / stats['total'] if stats['total'] > 0 else 0

        memory = self.engine.pathway_memory
        num_pathways = len(memory.transitions)
        num_attractors = len(memory.attractors)

        print(f"Session {stats['total']:4d} | "
              f"Pathways: {num_pathways:4d} | "
              f"Success: {success_rate:5.1f}% | "
              f"Snaps: {snap_rate:5.1f}% | "
              f"Avg: {avg_time:.3f}s/{avg_steps:.0f} steps")

    def print_final_report(self, stats: dict, elapsed: float):
        """Print final statistics."""
        success_rate = 100 * stats['successes'] / stats['total'] if stats['total'] > 0 else 0
        snap_rate = 100 * stats['snaps'] / stats['total'] if stats['total'] > 0 else 0
        avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
        avg_steps = stats['total_steps'] / stats['total'] if stats['total'] > 0 else 0

        memory = self.engine.pathway_memory
        num_pathways = len(memory.transitions)
        num_attractors = len(memory.attractors)

        # Find strongest pathways
        strong_pathways = sorted(
            [(f, t, s) for (f, t), stats in memory.transitions.items()
             for s in [stats.strength]],
            key=lambda x: x[2],
            reverse=True
        )[:10]

        print(f"\nTOTAL SESSIONS: {stats['total']}")
        print(f"TOTAL TIME: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"SESSIONS/SEC: {stats['total']/elapsed:.2f}")
        print()
        print(f"SUCCESS RATE: {success_rate:.1f}% ({stats['successes']}/{stats['total']})")
        print(f"SNAP RATE: {snap_rate:.1f}% ({stats['snaps']}/{stats['total']})")
        print(f"AVG TIME: {avg_time:.3f}s")
        print(f"AVG STEPS: {avg_steps:.1f}")
        print()
        print(f"PATHWAYS DISCOVERED: {num_pathways}")
        print(f"ATTRACTORS FOUND: {num_attractors}")
        print()

        if strong_pathways:
            print("STRONGEST PATHWAYS:")
            for i, (from_c, to_c, strength) in enumerate(strong_pathways, 1):
                stats_obj = memory.transitions[(from_c, to_c)]
                print(f"  {i:2d}. {from_c:20s} → {to_c:20s}  "
                      f"strength: {strength:.3f}  "
                      f"uses: {stats_obj.usage_count:3d}  "
                      f"success: {100*stats_obj.success_rate:.0f}%")


def main():
    """Run autonomous exploration."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous sphere explorer")
    parser.add_argument('--strategy', choices=['random', 'frontier', 'dense'],
                       default='frontier',
                       help='Exploration strategy')
    parser.add_argument('--sessions', type=int, default=None,
                       help='Max sessions (default: infinite)')
    parser.add_argument('--time', type=float, default=None,
                       help='Max time in seconds (default: infinite)')
    parser.add_argument('--report', type=int, default=10,
                       help='Report every N sessions (default: 10)')

    args = parser.parse_args()

    # Initialize engine
    print("Initializing generative learning engine...")
    engine = GenerativeLearningEngine()

    # Create explorer
    explorer = AutonomousExplorer(engine, exploration_strategy=args.strategy)

    # Run exploration
    explorer.explore_continuous(
        max_sessions=args.sessions,
        max_time=args.time,
        report_every=args.report
    )


if __name__ == '__main__':
    main()
