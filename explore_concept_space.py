#!/usr/bin/env python3
"""
Explore Concept Space
See how concepts are scored and related to each other
"""

import numpy as np
import json
from typing import List, Tuple
from primitive_pairing_generator import PrimitivePairingGenerator
from semantic_bloch_sphere import SemanticBlochSphere
from pathway_memory import PathwayMemory

def show_concept_details(name: str, sphere: SemanticBlochSphere, memory: PathwayMemory = None):
    """Show everything about a concept"""

    # Find concept
    concept = None
    for c in sphere.concepts:
        if c.name == name:
            concept = c
            break

    if not concept:
        print(f"Concept '{name}' not found")
        return

    print("="*80)
    print(f"CONCEPT: {name}")
    print("="*80)
    print()

    # Position
    print("POSITION ON BLOCH SPHERE:")
    print(f"  Coordinates: [{concept.coordinates[0]:.3f}, {concept.coordinates[1]:.3f}, {concept.coordinates[2]:.3f}]")

    # Convert to spherical
    r = np.linalg.norm(concept.coordinates)
    theta = np.arctan2(concept.coordinates[1], concept.coordinates[0])
    phi = np.arccos(np.clip(concept.coordinates[2] / r, -1, 1))

    print(f"  Radius: {r:.3f}")
    print(f"  Azimuth (θ): {np.rad2deg(theta):.1f}°")
    print(f"  Polar (φ): {np.rad2deg(phi):.1f}°")
    print()

    # Polarities
    print("POLARITY SIGNATURE:")
    for axis_name, value in concept.polarity_values.items():
        bar_length = int(abs(value) * 20)
        if value >= 0:
            bar = ' ' * 20 + '|' + '█' * bar_length
            label = f"{value:+.2f} "
        else:
            bar = '█' * bar_length + '|' + ' ' * 20
            label = f" {value:+.2f}"
        print(f"  {axis_name:15s} {bar} {label}")
    print()

    # Nearest neighbors
    print("NEAREST CONCEPTS (By Distance):")
    neighbors = sphere.find_nearest(concept, n=10)
    for i, (neighbor, distance) in enumerate(neighbors, 1):
        if neighbor.name == name:
            continue
        print(f"  {i}. {neighbor.name:20s} distance: {distance:.3f}")
    print()

    # Pathways (if memory provided)
    if memory:
        print("PATHWAYS FROM THIS CONCEPT:")
        outgoing = [(to_c, stats) for (from_c, to_c), stats in memory.transitions.items() if from_c == name]
        if outgoing:
            for to_c, stats in sorted(outgoing, key=lambda x: x[1].strength, reverse=True)[:5]:
                print(f"  → {to_c:20s} strength: {stats.strength:.3f}, uses: {stats.usage_count}, success: {stats.success_rate:.1%}")
        else:
            print("  (none)")
        print()

        print("PATHWAYS TO THIS CONCEPT:")
        incoming = [(from_c, stats) for (from_c, to_c), stats in memory.transitions.items() if to_c == name]
        if incoming:
            for from_c, stats in sorted(incoming, key=lambda x: x[1].strength, reverse=True)[:5]:
                print(f"  {from_c:20s} → strength: {stats.strength:.3f}, uses: {stats.usage_count}, success: {stats.success_rate:.1%}")
        else:
            print("  (none)")
        print()

        # Attractor status
        if name in memory.attractors:
            attractor = memory.attractors[name]
            print("ATTRACTOR STATUS:")
            print(f"  Visit count: {attractor.visit_count}")
            print(f"  Basin depth: {attractor.basin_depth:.3f}")
            print(f"  Stability: {attractor.stability_score:.3f}")
            print(f"  Harmony: {attractor.harmony_score:.3f}")
            print(f"  Low-order: {'Yes' if attractor.low_order else 'No'}")
            print()


def compare_concepts(name_a: str, name_b: str, sphere: SemanticBlochSphere):
    """Compare two concepts in detail"""

    # Find concepts
    concept_a = None
    concept_b = None
    for c in sphere.concepts:
        if c.name == name_a:
            concept_a = c
        if c.name == name_b:
            concept_b = c

    if not concept_a or not concept_b:
        print("One or both concepts not found")
        return

    print("="*80)
    print(f"COMPARISON: {name_a} ↔ {name_b}")
    print("="*80)
    print()

    # Distance
    distance = sphere.measure_similarity(concept_a, concept_b)
    print(f"GEOMETRIC DISTANCE: {distance:.3f}")
    print()

    # Position comparison
    print("POSITIONS:")
    print(f"  {name_a:20s} [{concept_a.coordinates[0]:+.3f}, {concept_a.coordinates[1]:+.3f}, {concept_a.coordinates[2]:+.3f}]")
    print(f"  {name_b:20s} [{concept_b.coordinates[0]:+.3f}, {concept_b.coordinates[1]:+.3f}, {concept_b.coordinates[2]:+.3f}]")
    print()

    # Polarity comparison
    print("POLARITY DIFFERENCES:")
    all_axes = set(concept_a.polarity_values.keys()) | set(concept_b.polarity_values.keys())
    for axis in sorted(all_axes):
        val_a = concept_a.polarity_values.get(axis, 0.0)
        val_b = concept_b.polarity_values.get(axis, 0.0)
        diff = val_b - val_a

        arrow = "→" if diff > 0 else "←" if diff < 0 else "="
        print(f"  {axis:15s} {val_a:+.2f} {arrow} {val_b:+.2f}  (Δ = {diff:+.2f})")
    print()

    # Phase relationship
    theta_a = np.arctan2(concept_a.coordinates[1], concept_a.coordinates[0])
    theta_b = np.arctan2(concept_b.coordinates[1], concept_b.coordinates[0])
    phase_diff = np.arctan2(np.sin(theta_b - theta_a), np.cos(theta_b - theta_a))

    print(f"PHASE RELATIONSHIP:")
    print(f"  {name_a} azimuth: {np.rad2deg(theta_a):.1f}°")
    print(f"  {name_b} azimuth: {np.rad2deg(theta_b):.1f}°")
    print(f"  Phase difference: {np.rad2deg(phase_diff):.1f}°")
    print()


def explore_neighborhood(name: str, sphere: SemanticBlochSphere, radius: float = 0.5):
    """Show all concepts within a radius"""

    # Find concept
    concept = None
    for c in sphere.concepts:
        if c.name == name:
            concept = c
            break

    if not concept:
        print(f"Concept '{name}' not found")
        return

    print("="*80)
    print(f"NEIGHBORHOOD: {name} (radius: {radius})")
    print("="*80)
    print()

    # Find neighbors within radius
    neighbors = []
    for c in sphere.concepts:
        if c.name == name:
            continue
        distance = sphere.measure_similarity(concept, c)
        if distance <= radius:
            neighbors.append((c.name, distance))

    neighbors.sort(key=lambda x: x[1])

    print(f"Found {len(neighbors)} concepts within radius {radius}:")
    print()

    for neighbor_name, distance in neighbors:
        # Show distance bar
        bar_length = int((1.0 - distance / radius) * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {neighbor_name:25s} [{bar}] {distance:.3f}")
    print()


def show_concept_cloud(sphere: SemanticBlochSphere, generation: int = None):
    """Show all concepts, optionally filtered by generation"""

    print("="*80)
    print("CONCEPT CLOUD")
    if generation is not None:
        print(f"Generation: {generation}")
    print("="*80)
    print()

    # Group by position (rough clustering)
    concepts = list(sphere.concepts)

    if generation is not None:
        # Filter by generation (would need pairing generator for this)
        print("(Generation filtering requires pairing generator)")

    # Sort by polar angle
    concepts.sort(key=lambda c: np.arctan2(c.coordinates[1], c.coordinates[0]))

    print(f"Total concepts: {len(concepts)}")
    print()

    print("Concepts by azimuth:")
    print()

    for i, concept in enumerate(concepts):
        theta = np.arctan2(concept.coordinates[1], concept.coordinates[0])
        phi = np.arccos(np.clip(concept.coordinates[2], -1, 1))

        # Visual indicator based on polar angle
        if phi < np.pi / 3:
            region = "NORTH"
            symbol = "▲"
        elif phi > 2 * np.pi / 3:
            region = "SOUTH"
            symbol = "▼"
        else:
            region = "EQUAT"
            symbol = "●"

        print(f"  {symbol} {concept.name:25s} θ:{np.rad2deg(theta):6.1f}° φ:{np.rad2deg(phi):6.1f}° {region}")

        if (i + 1) % 20 == 0:
            print()
    print()


def analyze_pathway_network(memory: PathwayMemory):
    """Analyze the pathway network structure"""

    print("="*80)
    print("PATHWAY NETWORK ANALYSIS")
    print("="*80)
    print()

    # Basic stats
    stats = memory.get_statistics()
    print("OVERALL STATISTICS:")
    print(f"  Total pathways: {stats['pathways']['total']}")
    print(f"  Strong pathways (>0.7): {stats['pathways']['strong']}")
    print(f"  Weak pathways (<0.3): {stats['pathways']['weak']}")
    print(f"  Total attractors: {stats['attractors']['total']}")
    print(f"  Cache accuracy: {stats['anticipation']['cache_accuracy']:.1%}")
    print()

    # Find hubs (concepts with many connections)
    out_degree = {}
    in_degree = {}

    for (from_c, to_c), stats_val in memory.transitions.items():
        out_degree[from_c] = out_degree.get(from_c, 0) + 1
        in_degree[to_c] = in_degree.get(to_c, 0) + 1

    print("TOP HUBS (Most Outgoing Pathways):")
    for concept, degree in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {concept:20s} → {degree} pathways")
    print()

    print("TOP ATTRACTORS (Most Incoming Pathways):")
    for concept, degree in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {concept:20s} ← {degree} pathways")
    print()

    # Strongest pathways
    print("STRONGEST PATHWAYS:")
    top_pathways = sorted(memory.transitions.items(), key=lambda x: x[1].strength, reverse=True)[:10]
    for (from_c, to_c), stats_val in top_pathways:
        print(f"  {from_c:15s} → {to_c:15s} strength: {stats_val.strength:.3f}, uses: {stats_val.usage_count}")
    print()


def main():
    """Interactive exploration"""

    print("="*80)
    print("CONCEPT SPACE EXPLORER")
    print("="*80)
    print()

    # Initialize system
    print("Initializing...")
    pairing_gen = PrimitivePairingGenerator(
        base_primitives=["compare", "interpret", "generate", "select"],
        max_generations=2
    )
    pairing_gen.generate_all()
    pairing_gen.map_to_semantic_sphere()
    sphere = pairing_gen.semantic_sphere

    # Try to load pathway memory
    try:
        memory = PathwayMemory()
        memory.load("learning_engine_state.json")
        print(f"✓ Loaded pathway memory ({len(memory.transitions)} pathways)")
    except:
        memory = None
        print("✓ No pathway memory loaded")

    print()

    # Examples
    print("EXAMPLE 1: Detailed concept view")
    print("-"*80)
    show_concept_details("understand", sphere, memory)

    print()
    print("EXAMPLE 2: Compare two concepts")
    print("-"*80)
    compare_concepts("compare", "interpret", sphere)

    print()
    print("EXAMPLE 3: Neighborhood exploration")
    print("-"*80)
    explore_neighborhood("understand", sphere, radius=0.3)

    print()
    print("EXAMPLE 4: Concept cloud overview")
    print("-"*80)
    show_concept_cloud(sphere)

    if memory:
        print()
        print("EXAMPLE 5: Pathway network analysis")
        print("-"*80)
        analyze_pathway_network(memory)

    print()
    print("="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print()
    print("Available commands:")
    print("  show <concept>              - Show concept details")
    print("  compare <concept_a> <concept_b> - Compare two concepts")
    print("  neighborhood <concept> [radius] - Explore neighborhood")
    print("  cloud                       - Show all concepts")
    print("  network                     - Analyze pathway network")
    print("  quit                        - Exit")
    print()

    while True:
        try:
            cmd = input("concept> ").strip()

            if not cmd:
                continue

            parts = cmd.split()
            command = parts[0].lower()

            if command == "quit":
                break

            elif command == "show" and len(parts) >= 2:
                show_concept_details(parts[1], sphere, memory)

            elif command == "compare" and len(parts) >= 3:
                compare_concepts(parts[1], parts[2], sphere)

            elif command == "neighborhood" and len(parts) >= 2:
                radius = float(parts[2]) if len(parts) >= 3 else 0.5
                explore_neighborhood(parts[1], sphere, radius)

            elif command == "cloud":
                show_concept_cloud(sphere)

            elif command == "network" and memory:
                analyze_pathway_network(memory)

            else:
                print("Unknown command. Type 'quit' to exit.")

            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

    print("Goodbye!")


if __name__ == "__main__":
    main()
