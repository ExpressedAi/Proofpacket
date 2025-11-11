"""
VBC Practical Demonstrations
============================

Real-world scenarios showing how the Variable Barrier Controller makes decisions
through phase-locked collapse with ε-gated hazard accumulation.

These are concrete examples demonstrating the universal framework in action.
"""

import numpy as np
from typing import List, Dict, Tuple
from vbc_prototype import VariableBarrierController, TokenCandidate, HourglassContext


# ====================================================================================
# SCENARIO 1: Restaurant Choice (Human Decision-Making)
# ====================================================================================

def demo_restaurant_choice():
    """
    Scenario: You're hungry and deciding between 5 restaurants.

    Constraints (Past cone):
    - Budget: $20
    - Time: 30 minutes
    - Preference: Italian food

    Options (Future cone):
    - A: Cheap pizza ($10, 10 min, Italian) ← high alignment, low brittleness
    - B: Expensive steakhouse ($50, 45 min, American) ← low alignment, high brittleness
    - C: Mid sushi ($25, 20 min, Japanese) ← medium everything
    - D: Fast food ($5, 5 min, Burgers) ← low alignment, low brittleness
    - E: Italian bistro ($18, 25 min, Italian) ← perfect fit!

    VBC should choose E (Italian bistro) due to high alignment + acceptable brittleness
    """

    print("=" * 70)
    print("SCENARIO 1: Restaurant Choice")
    print("=" * 70)
    print("\nContext: Hungry, $20 budget, 30 min available, want Italian\n")

    # Create mock "logits" representing desirability
    restaurants = {
        0: ("Cheap Pizza", 10, 10, "Italian", 0.7),      # (name, price, time, cuisine, base_appeal)
        1: ("Steakhouse", 50, 45, "American", 0.4),
        2: ("Sushi Place", 25, 20, "Japanese", 0.6),
        3: ("Fast Food", 5, 5, "Burgers", 0.3),
        4: ("Italian Bistro", 18, 25, "Italian", 0.9),
    }

    # Convert to logits (higher appeal → higher logit)
    logits = np.array([restaurants[i][4] for i in range(5)]) * 5  # Scale for softmax

    # Add noise for realism
    logits += np.random.randn(5) * 0.1

    vbc = VariableBarrierController(h_star=0.4, top_k=5)

    # Run VBC deliberation
    for tick in range(12):  # Max 3 cycles
        result = vbc.process_logits(logits)

        if result is not None:
            choice = restaurants[result.token_id]
            print(f"✓ Decision made after {tick+1} ticks!")
            print(f"  Choice: {choice[0]}")
            print(f"  Price: ${choice[1]}, Time: {choice[2]} min, Cuisine: {choice[3]}")
            print(f"  Hazard score: {result.hazard:.3f}")
            print(f"    ε={result.epsilon:.3f}, g={result.g_phi:.3f}, ζ={result.zeta:.3f}")
            print(f"    u={result.u:.3f}, p={result.p:.3f}")

            # Explain why this choice
            print(f"\n  Why this choice:")
            print(f"    - Alignment (u): {result.u:.3f} (fits Italian preference)")
            print(f"    - Brittleness (ζ): {result.zeta:.3f} (within budget & time)")
            print(f"    - Prior (p): {result.p:.3f} (high initial appeal)")
            break
    else:
        print("✗ No decision after 12 ticks (analysis paralysis!)")

    print()


# ====================================================================================
# SCENARIO 2: LLM Next-Token Prediction
# ====================================================================================

def demo_llm_token_selection():
    """
    Scenario: LLM generating text, needs to choose next token

    Prompt: "The capital of France is"

    Candidates:
    - "Paris" (p=0.92) ← correct, high probability
    - "London" (p=0.03) ← wrong but plausible
    - "the" (p=0.02) ← grammatically possible
    - "located" (p=0.01) ← verbose continuation
    - "..." (p=0.02) ← uncertainty marker

    VBC should choose "Paris" due to overwhelming probability + perfect alignment
    """

    print("=" * 70)
    print("SCENARIO 2: LLM Next-Token Selection")
    print("=" * 70)
    print('\nPrompt: "The capital of France is"\n')

    # Mock token probabilities (after softmax)
    tokens = {
        0: ("Paris", 0.92),
        1: ("London", 0.03),
        2: ("the", 0.02),
        3: ("located", 0.01),
        4: ("...", 0.02),
    }

    # Convert probabilities to logits
    probs = np.array([tokens[i][1] for i in range(5)])
    logits = np.log(probs + 1e-10) * 10  # Scale for numerical stability

    vbc = VariableBarrierController(h_star=0.3, top_k=5)

    # Run VBC
    for tick in range(8):
        result = vbc.process_logits(logits)

        if result is not None:
            choice = tokens[result.token_id]
            print(f"✓ Token committed after {tick+1} ticks!")
            print(f"  Selected token: '{choice[0]}'")
            print(f"  Probability: {choice[1]:.3f}")
            print(f"  Hazard score: {result.hazard:.3f}")
            print(f"    χ = {vbc.context.chi:.3f}")

            print(f"\n  Interpretation:")
            print(f"    - High probability (p={result.p:.3f}) → strong prior")
            print(f"    - Low entropy → high ε → easy capture")
            print(f"    - Perfect semantic fit → high u")
            print(f"    - System is supercritical (χ={vbc.context.chi:.3f}) → rapid commit")
            break
    else:
        print("✗ No commit (model is uncertain)")

    print()


# ====================================================================================
# SCENARIO 3: Trading Decision Under Uncertainty
# ====================================================================================

def demo_trading_decision():
    """
    Scenario: Trader deciding whether to buy, sell, or hold a stock

    Context:
    - Stock price: $100
    - Recent trend: +2% (bullish)
    - Market volatility: High (χ approaching 1)
    - Account balance: $10,000

    Options:
    - Buy: High risk, high reward (ζ high, u moderate)
    - Sell: Lock in gains (ζ low, u moderate)
    - Hold: Wait for more data (ζ low, u low)

    Under high volatility (χ → 1), VBC should prefer low-brittleness options (Sell or Hold)
    """

    print("=" * 70)
    print("SCENARIO 3: Trading Decision Under Market Stress")
    print("=" * 70)
    print("\nContext: Stock at $100, +2% trend, HIGH volatility (χ=0.9)\n")

    # Mock "logits" for each action
    actions = {
        0: ("Hold", 0.4, 0.2),    # (name, base_appeal, brittleness)
        1: ("Buy", 0.6, 0.8),     # High reward but high brittleness
        2: ("Sell", 0.5, 0.3),    # Moderate appeal, low brittleness
    }

    # Base logits from appeal
    logits = np.array([actions[i][1] for i in range(3)]) * 3
    logits += np.random.randn(3) * 0.1

    vbc = VariableBarrierController(h_star=0.35, top_k=3, zeta_star=0.7)

    # Simulate high market volatility by manually setting χ
    # (In real system, this would come from correlation analysis)

    print("Market state: HIGH VOLATILITY\n")

    for tick in range(12):
        result = vbc.process_logits(logits)

        if result is not None:
            choice = actions[result.token_id]
            print(f"✓ Decision after {tick+1} ticks: {choice[0].upper()}")
            print(f"  Hazard: {result.hazard:.3f}")
            print(f"  Brittleness: {result.zeta:.3f} / {vbc.zeta_star:.2f}")
            print(f"  Market χ: {vbc.context.chi:.3f}")

            print(f"\n  Analysis:")
            if choice[0] == "Buy":
                print(f"    - RISKY: High brittleness (ζ={result.zeta:.2f}) near limit!")
                print(f"    - Market stress → should prefer low-ζ options")
            elif choice[0] == "Sell":
                print(f"    - SAFE: Low brittleness (ζ={result.zeta:.2f})")
                print(f"    - Locks in gains, reduces exposure")
            else:  # Hold
                print(f"    - CAUTIOUS: Very low brittleness (ζ={result.zeta:.2f})")
                print(f"    - Wait for more data")

            break
    else:
        print("✗ Decision paralysis (χ too high → system frozen)")

    print()


# ====================================================================================
# SCENARIO 4: Multi-Option Reasoning (VBC Split/Join)
# ====================================================================================

def demo_multi_chain_reasoning():
    """
    Scenario: Solving a logic puzzle using multiple reasoning chains

    Problem: "If all roses are flowers, and some flowers fade quickly,
              do all roses fade quickly?"

    Strategy: Split into 3 reasoning chains:
    1. Formal logic chain (deductive)
    2. Counterexample chain (find exception)
    3. Probabilistic chain (base rates)

    Join by weighted hazard → choose most confident answer
    """

    print("=" * 70)
    print("SCENARIO 4: Multi-Chain Reasoning")
    print("=" * 70)
    print('\nPuzzle: "If all roses are flowers, and some flowers fade quickly,')
    print('         do all roses fade quickly?"\n')

    # Three reasoning approaches
    approaches = {
        "deductive": {
            "answer": "No (invalid syllogism)",
            "confidence": 0.9,
            "reasoning": "Some ≠ All; cannot conclude"
        },
        "counterexample": {
            "answer": "No (preserved roses exist)",
            "confidence": 0.8,
            "reasoning": "Found counterexample"
        },
        "probabilistic": {
            "answer": "Probably no",
            "confidence": 0.6,
            "reasoning": "Base rate argument"
        }
    }

    # Create separate VBC instances for each chain
    chains = {}

    for name, approach in approaches.items():
        vbc = VariableBarrierController(h_star=0.4, top_k=2)

        # Create logits favoring this approach's answer
        if "No" in approach["answer"]:
            logits = np.array([approach["confidence"] * 5, (1-approach["confidence"]) * 5])
        else:
            logits = np.array([(1-approach["confidence"]) * 5, approach["confidence"] * 5])

        # Run chain until commit
        for tick in range(12):
            result = vbc.process_logits(logits)
            if result is not None:
                chains[name] = {
                    "result": result,
                    "answer": approach["answer"],
                    "reasoning": approach["reasoning"],
                    "hazard": result.hazard,
                    "ticks": tick + 1
                }
                break

    # Display all chains
    print("Reasoning chains:\n")
    for name, chain in chains.items():
        print(f"  {name.upper()}:")
        print(f"    Answer: {chain['answer']}")
        print(f"    Reasoning: {chain['reasoning']}")
        print(f"    Hazard: {chain['hazard']:.3f}")
        print(f"    Ticks: {chain['ticks']}")
        print()

    # Join by selecting highest hazard
    winner = max(chains.items(), key=lambda x: x[1]['hazard'])

    print(f"✓ JOIN: Selected chain = {winner[0].upper()}")
    print(f"  Final answer: {winner[1]['answer']}")
    print(f"  Confidence (hazard): {winner[1]['hazard']:.3f}")
    print(f"\n  Why: Highest phase-lock strength (hazard) wins")
    print()


# ====================================================================================
# SCENARIO 5: Freezing Under Pressure
# ====================================================================================

def demo_freezing_under_pressure():
    """
    Scenario: Student taking timed exam, running out of time

    Question: "What is the capital of Slovenia?"

    Candidates:
    - Ljubljana (correct)
    - Zagreb (Croatia's capital - common confusion)
    - Vienna (nearby, Austria)
    - Don't know

    As time pressure increases → ζ → ζ* → all hazards → 0 → FREEZE
    """

    print("=" * 70)
    print("SCENARIO 5: Freezing Under Pressure")
    print("=" * 70)
    print('\nQuestion: "What is the capital of Slovenia?"\n')

    # Student's knowledge (uncertain)
    answers = {
        0: ("Ljubljana", 0.4),  # Correct but not confident
        1: ("Zagreb", 0.3),     # Common confusion
        2: ("Vienna", 0.1),     # Wrong
        3: ("Skip", 0.2),       # Give up
    }

    logits = np.array([answers[i][1] for i in range(4)]) * 3

    # Simulate increasing time pressure
    time_pressures = [
        ("RELAXED (5 min left)", 0.3),
        ("MODERATE (2 min left)", 0.6),
        ("HIGH (30 sec left)", 0.85),
        ("EXTREME (10 sec left)", 0.95),
    ]

    for pressure_name, zeta_pressure in time_pressures:
        print(f"{pressure_name}:")

        # Create VBC with zeta_star that will be hit by time pressure
        vbc = VariableBarrierController(h_star=0.3, top_k=4, zeta_star=1.0)

        # Override brittleness to reflect time pressure
        # (In real system, this would come from cognitive load measurement)
        original_compute_brittleness = vbc._compute_brittleness

        def pressured_brittleness(candidate):
            base = original_compute_brittleness(candidate)
            return min(1.0, base + zeta_pressure)  # Add time pressure

        vbc._compute_brittleness = pressured_brittleness

        # Try to make decision
        committed = False
        for tick in range(12):
            result = vbc.process_logits(logits)
            if result is not None:
                choice = answers[result.token_id]
                print(f"  ✓ Answered: '{choice[0]}' (hazard={result.hazard:.3f}, ζ={result.zeta:.3f})")
                committed = True
                break

        if not committed:
            print(f"  ✗ FROZEN - Cannot decide! (brittleness too high)")
            print(f"     → Term (1 - ζ/ζ*) ≈ 0 → all hazards ≈ 0")

        print()

    print("Interpretation:")
    print("  - Low pressure: Can think, makes decision")
    print("  - High pressure: ζ → ζ* → hazard collapses → FREEZE")
    print("  - This explains 'choking under pressure' mathematically!")
    print()


# ====================================================================================
# RUN ALL DEMOS
# ====================================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "VBC PRACTICAL DEMONSTRATIONS" + " " * 25 + "║")
    print("║" + " " * 12 + "Phase-Locked Decision Making in Action" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    demo_restaurant_choice()
    demo_llm_token_selection()
    demo_trading_decision()
    demo_multi_chain_reasoning()
    demo_freezing_under_pressure()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll scenarios demonstrate the same universal mechanism:")
    print("  h = κ · ε · g(e_φ) · (1 - ζ/ζ*) · u · p")
    print("\nKey insights:")
    print("  1. Decisions happen when hazard h exceeds threshold h*")
    print("  2. Brittleness ζ → ζ* causes freezing (pressure, complexity)")
    print("  3. High χ (criticality) → rapid collapse or paralysis")
    print("  4. Low-order preference: simple options win when uncertain")
    print("  5. Multi-chain reasoning: highest hazard wins the join")
    print("\nThis is NOT different algorithms for different domains.")
    print("It's ONE algorithm working across all substrates.")
    print("\n" + "=" * 70 + "\n")
