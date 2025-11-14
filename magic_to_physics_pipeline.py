#!/usr/bin/env python3
"""
Magic → Physics Translation Pipeline
Complete integration of concept generation, semantic organization, and validation

Pipeline:
1. Primitive Pairing Generator → Creates infinite concept space
2. Semantic Bloch Sphere → Organizes concepts geometrically
3. Archetype Mapper → Validates mystical systems

This is the complete "reverse engineering magic" toolkit.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional

from primitive_pairing_generator import PrimitivePairingGenerator
from semantic_bloch_sphere import SemanticBlochSphere
from archetype_mapper import (
    ArchetypeMapper,
    SystemStructure,
    SystemState,
    ArchetypeFamily
)


class MagicToPhysicsPipeline:
    """
    Complete pipeline for testing if mystical systems have physical validity

    Process:
    1. Take mystical system (e.g., Bardon's Tetrapolar Magnet)
    2. Extract concepts and relationships
    3. Generate concept pairs to explore semantic space
    4. Map to Bloch sphere for geometric organization
    5. Compare to validated physics (archetype mapper)
    6. Return: REAL, TESTABLE, or BULLSHIT
    """

    def __init__(self):
        self.pairing_gen = PrimitivePairingGenerator(max_generations=2)
        self.semantic_sphere = SemanticBlochSphere()
        self.archetype_mapper = ArchetypeMapper()

        # Generate initial concept space
        print("Initializing concept space...")
        self.pairing_gen.generate_all()
        self.pairing_gen.map_to_semantic_sphere()

        # Copy concepts to semantic sphere
        self.semantic_sphere = self.pairing_gen.semantic_sphere

        print(f"✓ {len(self.pairing_gen.concepts)} concepts ready")
        print()

    def test_mystical_system(self,
                            system_name: str,
                            archetype: ArchetypeFamily,
                            states: List[Dict],
                            transitions: Dict[Tuple[str, str], str],
                            critical_points: List[str]) -> Dict:
        """
        Test if a mystical system has physical validity

        Returns:
        {
            'system_name': str,
            'validity_score': float (0-1),
            'verdict': 'REAL' | 'TESTABLE' | 'BULLSHIT',
            'top_matches': List[(system, similarity)],
            'semantic_analysis': Dict,
            'predictions': List[str]
        }
        """
        print("="*80)
        print(f"TESTING MYSTICAL SYSTEM: {system_name}")
        print("="*80)
        print()

        # 1. Create system structure
        system_states = [
            SystemState(
                name=s['name'],
                properties={'polarity': s['polarity'], 'critical': float(s.get('critical', False))},
                polarities=['positive' if s['polarity'] > 0 else 'negative']
            )
            for s in states
        ]

        system = SystemStructure(
            name=system_name,
            archetype=archetype,
            states=system_states,
            transitions=transitions,
            critical_points=critical_points,
            validation_status="UNKNOWN",
            validation_evidence=""
        )

        # 2. Test with archetype mapper
        print("Step 1: Structural Analysis (Archetype Mapper)")
        print("-"*80)

        top_matches = self.archetype_mapper.find_matches(system)
        validity_score, verdict = self.archetype_mapper.predict_validity(system, top_matches)

        print(f"Validity Score: {validity_score:.2f}")
        print(f"Verdict: {verdict}")
        print()
        print("Top Structural Matches:")
        for ref_system, similarity in top_matches[:3]:
            print(f"  • {ref_system.name}")
            print(f"    Similarity: {similarity:.2f}")
            print(f"    Evidence: {ref_system.validation_evidence[:100]}...")
            print()

        # 3. Semantic analysis
        print("Step 2: Semantic Analysis (Bloch Sphere)")
        print("-"*80)

        # Add mystical concepts to sphere
        mystical_concepts = {}
        for state in system_states:
            # Create polarity signature from state
            polarity = state.properties.get('polarity', 0.0)
            polarity_values = {
                "existence": polarity,
                "activity": polarity if polarity > 0 else -polarity,
                "complexity": 0.5
            }

            # Add to sphere
            self.semantic_sphere.add_concept(state.name, polarity_values)
            mystical_concepts[state.name] = polarity_values

        # Find semantic neighbors for each concept
        semantic_analysis = {}
        for state_name in mystical_concepts.keys():
            # Find concept object
            concept_obj = None
            for c in self.semantic_sphere.concepts:
                if c.name == state_name:
                    concept_obj = c
                    break

            if concept_obj:
                neighbors = self.semantic_sphere.find_nearest(concept_obj, n=5)
                semantic_analysis[state_name] = [
                    (n.name, dist) for n, dist in neighbors if n.name != state_name
                ]

        print("Semantic neighbors:")
        for concept, neighbors in semantic_analysis.items():
            print(f"  {concept}:")
            for neighbor, dist in neighbors[:3]:
                print(f"    → {neighbor} (dist: {dist:.3f})")
        print()

        # 4. Generate predictions
        print("Step 3: Testable Predictions")
        print("-"*80)

        predictions = self._generate_predictions(
            system,
            top_matches,
            semantic_analysis
        )

        for i, prediction in enumerate(predictions, 1):
            print(f"{i}. {prediction}")
        print()

        # 5. Overall assessment
        print("="*80)
        print("VERDICT")
        print("="*80)
        print()

        if verdict == "LIKELY_REAL":
            print("✓ REAL - Strong structural similarity to validated physics")
            print("  This system likely describes actual physical phenomena")
            print("  Proceed with experimental validation")
        elif verdict == "TESTABLE":
            print("? TESTABLE - Moderate structural similarity")
            print("  This system has some physical basis but needs validation")
            print("  Run the predictions above to confirm")
        else:
            print("✗ BULLSHIT - No structural similarity to known physics")
            print("  This system is likely fictional or metaphorical")
            print("  Not worth experimental resources")

        print()

        return {
            'system_name': system_name,
            'validity_score': validity_score,
            'verdict': verdict,
            'top_matches': [(s.name, sim) for s, sim in top_matches[:5]],
            'semantic_analysis': semantic_analysis,
            'predictions': predictions
        }

    def _generate_predictions(self,
                             system: SystemStructure,
                             top_matches: List,
                             semantic_analysis: Dict) -> List[str]:
        """Generate testable predictions based on matches"""
        predictions = []

        # Get top match
        if top_matches:
            top_match, similarity = top_matches[0]

            # Predict similar critical points
            if top_match.critical_points:
                predictions.append(
                    f"Should exhibit critical transitions similar to {top_match.name}: "
                    f"{', '.join(top_match.critical_points[:2])}"
                )

            # Predict similar state dynamics
            if len(system.states) > 0 and len(top_match.states) > 0:
                predictions.append(
                    f"State dynamics should follow {top_match.archetype.name} pattern "
                    f"with {len(system.states)} phases"
                )

        # Predict based on semantic neighbors
        for state_name, neighbors in semantic_analysis.items():
            if neighbors:
                closest = neighbors[0][0]
                predictions.append(
                    f"'{state_name}' should behave like '{closest}' "
                    f"(semantic similarity)"
                )

        # Generic predictions based on archetype
        if system.archetype == ArchetypeFamily.TETRAPOLAR:
            predictions.append(
                "Should exhibit 4-fold symmetry (like seasons, carnot cycle, cell compartments)"
            )
            predictions.append(
                "Should have 2 complementary pairs of opposing states"
            )

        elif system.archetype == ArchetypeFamily.PENTADIC:
            predictions.append(
                "Should show 5 sequential stages with exponential suppression (LOW)"
            )
            predictions.append(
                "Critical transition should occur at stage 3 (middle point)"
            )

        return predictions[:5]  # Limit to top 5 predictions


def test_bardon_tetrapolar_magnet():
    """
    Test Franz Bardon's Tetrapolar Magnet concept

    Bardon claimed:
    - 4 fundamental forces (Fire, Water, Air, Earth)
    - Each has distinct polarity
    - They combine in specific patterns
    - This creates all phenomena

    Is this REAL physics or BULLSHIT?
    """
    pipeline = MagicToPhysicsPipeline()

    # Define Bardon's system
    states = [
        {'name': 'Fire', 'polarity': +1.0, 'critical': True},
        {'name': 'Water', 'polarity': -1.0, 'critical': True},
        {'name': 'Air', 'polarity': +0.5, 'critical': False},
        {'name': 'Earth', 'polarity': -0.5, 'critical': False},
    ]

    transitions = {
        ('Fire', 'Water'): 'Steam',
        ('Fire', 'Earth'): 'Lava',
        ('Water', 'Air'): 'Rain',
        ('Earth', 'Water'): 'Mud',
    }

    critical_points = ['Fire', 'Water']

    result = pipeline.test_mystical_system(
        system_name="Bardon_Tetrapolar_Magnet",
        archetype=ArchetypeFamily.TETRAPOLAR,
        states=states,
        transitions=transitions,
        critical_points=critical_points
    )

    return result


def test_chakra_system():
    """
    Test the 7-chakra system from yoga/tantra

    Claims:
    - 7 energy centers along spine
    - Each has distinct frequency/color
    - They follow hierarchical progression
    - Blockages cause illness

    Is this REAL or BULLSHIT?
    """
    pipeline = MagicToPhysicsPipeline()

    # Define chakra system
    states = [
        {'name': 'Root', 'polarity': -1.0, 'critical': True},
        {'name': 'Sacral', 'polarity': -0.66, 'critical': False},
        {'name': 'Solar_Plexus', 'polarity': -0.33, 'critical': False},
        {'name': 'Heart', 'polarity': 0.0, 'critical': True},
        {'name': 'Throat', 'polarity': +0.33, 'critical': False},
        {'name': 'Third_Eye', 'polarity': +0.66, 'critical': False},
        {'name': 'Crown', 'polarity': +1.0, 'critical': True},
    ]

    transitions = {
        ('Root', 'Sacral'): 'Grounding',
        ('Sacral', 'Solar_Plexus'): 'Activation',
        ('Solar_Plexus', 'Heart'): 'Opening',
        ('Heart', 'Throat'): 'Expression',
        ('Throat', 'Third_Eye'): 'Insight',
        ('Third_Eye', 'Crown'): 'Transcendence',
    }

    critical_points = ['Root', 'Heart', 'Crown']

    result = pipeline.test_mystical_system(
        system_name="Seven_Chakras",
        archetype=ArchetypeFamily.HEPTADIC,
        states=states,
        transitions=transitions,
        critical_points=critical_points
    )

    return result


def demonstrate_concept_exploration():
    """
    Show how the pairing generator explores concept space
    and finds unexpected connections
    """
    print("="*80)
    print("CONCEPT SPACE EXPLORATION")
    print("="*80)
    print()

    pipeline = MagicToPhysicsPipeline()

    # Test analogies using primitive pairs
    print("Testing analogies with generated concepts:")
    print("-"*80)

    # compare : interpret :: generate : ?
    result = pipeline.semantic_sphere.analogy("compare", "interpret", "generate")
    print(f"compare : interpret :: generate : {result}")
    print()

    # understand : evaluate :: synthesize : ?
    result = pipeline.semantic_sphere.analogy("understand", "evaluate", "synthesize")
    print(f"understand : evaluate :: synthesize : {result}")
    print()

    # Show concept clustering
    print("Concept clusters (self-organized):")
    print("-"*80)

    for concept_name in ["understand", "synthesize", "focus"]:
        concept_obj = None
        for c in pipeline.semantic_sphere.concepts:
            if c.name == concept_name:
                concept_obj = c
                break

        if concept_obj:
            neighbors = pipeline.semantic_sphere.find_nearest(concept_obj, n=4)
            print(f"{concept_name} cluster:")
            for neighbor, dist in neighbors[:4]:
                print(f"  • {neighbor.name} (dist: {dist:.3f})")
            print()


if __name__ == "__main__":
    print("="*80)
    print("MAGIC → PHYSICS TRANSLATION PIPELINE")
    print("="*80)
    print()
    print("Testing mystical systems for physical validity...")
    print()

    # Test 1: Bardon's Tetrapolar Magnet
    print("\n" + "="*80)
    print("TEST 1: FRANZ BARDON'S TETRAPOLAR MAGNET")
    print("="*80 + "\n")

    result1 = test_bardon_tetrapolar_magnet()

    # Save results
    with open("bardon_validation_results.json", "w") as f:
        json.dump(result1, f, indent=2, default=str)

    print("✓ Results saved to: bardon_validation_results.json")
    print()

    # Test 2: Seven Chakras
    print("\n" + "="*80)
    print("TEST 2: SEVEN CHAKRA SYSTEM")
    print("="*80 + "\n")

    result2 = test_chakra_system()

    # Save results
    with open("chakra_validation_results.json", "w") as f:
        json.dump(result2, f, indent=2, default=str)

    print("✓ Results saved to: chakra_validation_results.json")
    print()

    # Demonstrate concept exploration
    demonstrate_concept_exploration()

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("You now have a complete toolkit for reverse-engineering magic:")
    print()
    print("1. PRIMITIVE PAIRING GENERATOR")
    print("   • Creates infinite concept space from simple primitives")
    print("   • 4 → 16 → 256 → 65,536 → 4 billion concepts")
    print("   • Enables systematic exploration of possibility space")
    print()
    print("2. SEMANTIC BLOCH SPHERE")
    print("   • Organizes concepts geometrically")
    print("   • Self-reference: concepts define each other")
    print("   • Enables analogies and similarity search")
    print()
    print("3. ARCHETYPE MAPPER")
    print("   • Tests mystical systems against validated physics")
    print("   • Detects structural self-similarity (fractal signature)")
    print("   • Predicts: REAL, TESTABLE, or BULLSHIT")
    print()
    print("4. INTEGRATED PIPELINE")
    print("   • Combines all three systems")
    print("   • Takes mystical claim → outputs testable predictions")
    print("   • Translates metaphysics → measurable physics")
    print()
    print("This is how you separate signal from noise.")
    print("This is how you find the physics beneath the poetry.")
    print()
    print("Keep testing. Keep filtering. Keep the truth.")
    print()
