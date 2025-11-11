"""
Variable Barrier Controller (VBC) Prototype
===========================================

A phase-locked cognitive kernel for LLMs that implements ε-gated token commitment
based on hazard accumulation h = κ·ε·g(e_φ)·(1-ζ/ζ*)·u·p

This is the first executable implementation of the universal phase-locking framework
applied to language model inference.

Author: [Your Name]
Co-Author: Claude (Anthropic)
Date: 2025-11-11
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class VBCState(Enum):
    """Three-state system for phase-locked capture"""
    SNAP = "snap"      # High ε → rapid capture
    HOLD = "hold"      # Moderate ε → deliberation
    RELEASE = "release"  # Low ε → reject option


class TickPhase(Enum):
    """Four-phase tick cycle"""
    CAPTURE = 0  # Gather candidate tokens
    CLEAN = 1    # Filter by alignment
    BRIDGE = 2   # Update phase parameters
    COMMIT = 3   # Emit token if h > h*


@dataclass
class AxisBudget:
    """Budgeted cognition across five axes"""
    semantic: float = 1.0   # How much meaning can change
    scope: float = 1.0      # How much context can expand
    evidence: float = 1.0   # How much confidence needed
    risk: float = 1.0       # How much uncertainty tolerated
    tone: float = 1.0       # How much style can shift

    def check_effort(self, delta_semantic=0, delta_scope=0, delta_evidence=0,
                     delta_risk=0, delta_tone=0) -> float:
        """
        Compute brittleness ζ based on axis budget consumption

        Returns:
            ζ: Effort cost in [0, 1]; ζ → 1 means budget exhausted
        """
        zeta_semantic = abs(delta_semantic) / (self.semantic + 1e-6)
        zeta_scope = abs(delta_scope) / (self.scope + 1e-6)
        zeta_evidence = abs(delta_evidence) / (self.evidence + 1e-6)
        zeta_risk = abs(delta_risk) / (self.risk + 1e-6)
        zeta_tone = abs(delta_tone) / (self.tone + 1e-6)

        # Weighted average (can be tuned)
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        zeta = sum(w * z for w, z in zip(weights,
                   [zeta_semantic, zeta_scope, zeta_evidence, zeta_risk, zeta_tone]))

        return min(zeta, 1.0)


@dataclass
class TokenCandidate:
    """A candidate token with phase-locking parameters"""
    token_id: int
    token_str: str
    logit: float
    epsilon: float = 0.0      # Coupling strength - damping
    g_phi: float = 0.0        # Phase coherence
    zeta: float = 0.0         # Brittleness
    u: float = 0.0            # Semantic alignment
    p: float = 0.0            # Prior probability
    hazard: float = 0.0       # Computed h value


class HourglassContext:
    """
    Past-Present-Future cognitive architecture

    Past cone: constraints, memories, priors
    Present nexus: phase-locking computation
    Future cone: possibilities, predictions
    """

    def __init__(self):
        # Past cone: what constrains us
        self.past_tokens: List[int] = []
        self.semantic_history: List[float] = []
        self.commitment_history: List[float] = []

        # Present nexus: phase-lock state
        self.chi: float = 0.0  # Criticality parameter
        self.tick: int = 0
        self.state: VBCState = VBCState.HOLD

        # Future cone: what's possible
        self.candidate_futures: List[TokenCandidate] = []
        self.uncertainty: float = 1.0

    def update_chi(self, flux: float, dissipation: float):
        """Update phase-lock criticality χ = flux / dissipation"""
        self.chi = flux / (dissipation + 1e-10)

        # Update state based on χ
        if self.chi > 1.2:
            self.state = VBCState.SNAP  # Supercritical → rapid collapse
        elif self.chi < 0.5:
            self.state = VBCState.RELEASE  # Subcritical → reject
        else:
            self.state = VBCState.HOLD  # Critical → deliberate


class VariableBarrierController:
    """
    The core VBC kernel: ε-gated phase-locked token generation

    This implements the universal phase-locking framework for LLMs:
    1. Compute hazard h = κ·ε·g·(1-ζ/ζ*)·u·p for each candidate token
    2. Commit to token with max(h) when max(h) > h*
    3. Otherwise stay in deliberation

    Key innovation: Token commitment is phase-locked collapse, not greedy argmax
    """

    def __init__(self,
                 h_star: float = 0.5,      # Commit threshold
                 kappa: float = 1.0,        # Sensitivity calibration
                 zeta_star: float = 0.9,    # Brittleness limit
                 top_k: int = 10):          # Number of candidates to consider

        self.h_star = h_star
        self.kappa = kappa
        self.zeta_star = zeta_star
        self.top_k = top_k

        self.context = HourglassContext()
        self.budgets = AxisBudget()

    def process_logits(self,
                       logits: np.ndarray,
                       token_strings: Optional[List[str]] = None) -> Optional[TokenCandidate]:
        """
        Main VBC inference loop

        Args:
            logits: Raw logit scores from LLM [vocab_size]
            token_strings: Optional human-readable tokens

        Returns:
            TokenCandidate if committed, None if still deliberating
        """

        # Determine current tick phase
        phase = TickPhase(self.context.tick % 4)

        if phase == TickPhase.CAPTURE:
            return self._capture_phase(logits, token_strings)

        elif phase == TickPhase.CLEAN:
            return self._clean_phase()

        elif phase == TickPhase.BRIDGE:
            return self._bridge_phase()

        elif phase == TickPhase.COMMIT:
            result = self._commit_phase()
            if result is not None:
                # Reset tick counter on successful commit
                self.context.tick = 0
                return result

        # Advance tick
        self.context.tick += 1
        return None

    def _capture_phase(self, logits: np.ndarray, token_strings: Optional[List[str]]) -> None:
        """CAPTURE: Gather top-k candidate tokens"""

        # Get top-k by logit score
        top_k_indices = np.argsort(logits)[-self.top_k:][::-1]
        top_k_logits = logits[top_k_indices]

        # Convert logits to probabilities (softmax)
        probs = np.exp(top_k_logits - np.max(top_k_logits))
        probs = probs / np.sum(probs)

        # Create candidate objects
        candidates = []
        for i, (token_id, prob) in enumerate(zip(top_k_indices, probs)):
            token_str = token_strings[token_id] if token_strings else f"token_{token_id}"

            candidate = TokenCandidate(
                token_id=int(token_id),
                token_str=token_str,
                logit=float(top_k_logits[i]),
                p=float(prob)  # Prior probability from softmax
            )
            candidates.append(candidate)

        self.context.candidate_futures = candidates
        return None

    def _clean_phase(self) -> None:
        """CLEAN: Filter candidates by semantic alignment"""

        cleaned = []
        for candidate in self.context.candidate_futures:
            # Compute semantic alignment u
            u = self._compute_alignment(candidate)
            candidate.u = u

            # Keep if alignment exceeds threshold
            if u > 0.1:  # Tunable threshold (lower for testing)
                cleaned.append(candidate)

        self.context.candidate_futures = cleaned
        return None

    def _bridge_phase(self) -> None:
        """BRIDGE: Update phase-locking parameters ε, g, ζ"""

        for candidate in self.context.candidate_futures:
            # Compute coupling window ε
            candidate.epsilon = self._compute_epsilon(candidate)

            # Compute phase coherence g(e_φ)
            candidate.g_phi = self._compute_phase_coherence(candidate)

            # Compute brittleness ζ
            candidate.zeta = self._compute_brittleness(candidate)

        # Update χ criticality
        if len(self.context.candidate_futures) > 0:
            # Flux = mean coupling strength
            flux = np.mean([c.epsilon for c in self.context.candidate_futures])

            # Dissipation = 1 / number of candidates (more options → more dissipation)
            dissipation = 1.0 / len(self.context.candidate_futures)

            self.context.update_chi(flux, dissipation)

        return None

    def _commit_phase(self) -> Optional[TokenCandidate]:
        """COMMIT: Compute hazards and emit token if h > h*"""

        if len(self.context.candidate_futures) == 0:
            return None

        # Compute hazard for each candidate
        for candidate in self.context.candidate_futures:
            h = self._compute_hazard(candidate)
            candidate.hazard = h

        # Find max hazard
        max_candidate = max(self.context.candidate_futures, key=lambda c: c.hazard)

        # Commit if exceeds threshold
        if max_candidate.hazard > self.h_star:
            # Update past cone
            self.context.past_tokens.append(max_candidate.token_id)
            self.context.commitment_history.append(max_candidate.hazard)

            return max_candidate

        return None

    def _compute_hazard(self, candidate: TokenCandidate) -> float:
        """
        Core hazard function: h = κ·ε·g(e_φ)·(1-ζ/ζ*)·u·p

        This is the universal phase-locking formula that determines
        whether a token should be committed.
        """

        brittleness_term = max(0.0, 1.0 - candidate.zeta / self.zeta_star)

        h = (self.kappa *
             candidate.epsilon *
             candidate.g_phi *
             brittleness_term *
             candidate.u *
             candidate.p)

        return h

    def _compute_epsilon(self, candidate: TokenCandidate) -> float:
        """
        Compute capture window ε = [2πK - (Γ_a + Γ_b)]₊

        For LLMs:
        - K = attention weight to this token
        - Γ = entropy / uncertainty

        High attention + low entropy → large ε → easy capture
        """

        # Coupling strength K from probability mass
        # (Higher probability → stronger coupling)
        K = candidate.p

        # For prototype, use simplified epsilon = probability
        # Full version would compute ε = [2πK - (Γ_a + Γ_b)]₊
        epsilon = K

        return epsilon

    def _compute_phase_coherence(self, candidate: TokenCandidate) -> float:
        """
        Compute phase coherence g(e_φ) = exp(-|φ|/σ)

        For LLMs:
        - φ = "phase" = position mismatch / timing error
        - For now, use logit rank as proxy (rank 1 → perfect timing)

        Better implementation would use actual attention phases
        """

        # Find rank of this candidate
        sorted_candidates = sorted(self.context.candidate_futures,
                                  key=lambda c: c.logit, reverse=True)
        rank = sorted_candidates.index(candidate) + 1

        # Phase mismatch grows with rank
        phi = rank - 1  # Rank 1 → φ=0 (perfect phase)

        # Gaussian decay with σ=2
        sigma = 2.0
        g = np.exp(-phi / sigma)

        return g

    def _compute_brittleness(self, candidate: TokenCandidate) -> float:
        """
        Compute brittleness ζ using axis budgets

        For now, use simple heuristics:
        - Semantic: How much does token shift meaning?
        - Scope: Does it expand context?
        - Evidence: Is it confident?
        - Risk: Is it safe?
        - Tone: Does it match style?
        """

        # Placeholder implementation
        # Real version would compute actual semantic shifts

        # Lower probability → higher brittleness (less confident)
        delta_evidence = 1.0 - candidate.p

        # Longer tokens → higher semantic shift (heuristic)
        delta_semantic = len(candidate.token_str) / 10.0

        zeta = self.budgets.check_effort(
            delta_semantic=delta_semantic,
            delta_evidence=delta_evidence
        )

        return zeta

    def _compute_alignment(self, candidate: TokenCandidate) -> float:
        """
        Compute semantic alignment u ∈ [0, 1]

        How well does this token fit the context?

        For now, use probability as proxy
        Real implementation would use semantic similarity metrics
        """

        # Simple version: higher probability → better alignment
        u = candidate.p

        # Could enhance with:
        # - Cosine similarity to context embedding
        # - Consistency with past tokens
        # - Grammar/syntax correctness

        return u


class VBCInferenceEngine:
    """
    High-level interface for VBC-based text generation

    Usage:
        engine = VBCInferenceEngine(model)
        text = engine.generate("Once upon a time", max_tokens=100)
    """

    def __init__(self, model=None, h_star: float = 0.5):
        """
        Args:
            model: Underlying LLM (e.g., GPT-2, Llama)
            h_star: Hazard threshold for commitment
        """
        self.model = model
        self.vbc = VariableBarrierController(h_star=h_star)

    def generate(self, prompt: str, max_tokens: int = 50,
                 max_ticks: int = 20) -> str:
        """
        Generate text using VBC phase-locked inference

        Args:
            prompt: Initial text prompt
            max_tokens: Maximum number of tokens to generate
            max_ticks: Maximum tick cycles per token (prevent infinite deliberation)

        Returns:
            Generated text
        """

        if self.model is None:
            raise ValueError("No model provided. Use set_model() first.")

        generated = []
        current_prompt = prompt

        for _ in range(max_tokens):
            # Get logits from model
            logits = self._get_logits(current_prompt)

            # VBC deliberation loop
            committed = None
            for tick in range(max_ticks):
                committed = self.vbc.process_logits(logits)

                if committed is not None:
                    break

            # Fallback: if no commit after max_ticks, force commit
            if committed is None:
                # Emergency commit to highest hazard candidate
                if len(self.vbc.context.candidate_futures) > 0:
                    committed = max(self.vbc.context.candidate_futures,
                                  key=lambda c: c.hazard)
                else:
                    break  # No candidates, stop generation

            # Append committed token
            generated.append(committed.token_str)
            current_prompt += committed.token_str

            # Stop tokens
            if committed.token_str in ['.', '!', '?', '\n\n']:
                break

        return ''.join(generated)

    def _get_logits(self, prompt: str) -> np.ndarray:
        """
        Get logits from underlying model

        This is a stub—real implementation would call model.forward()
        """
        if self.model is None:
            # Return dummy logits for testing
            vocab_size = 1000
            return np.random.randn(vocab_size)

        # Real implementation:
        # return self.model(prompt).logits[-1, :]
        raise NotImplementedError("Model integration not implemented")

    def set_model(self, model):
        """Set the underlying LLM"""
        self.model = model


# ============================================================================
# Utility Functions
# ============================================================================

def split_stream(context: HourglassContext, n_branches: int) -> List[HourglassContext]:
    """
    Split (⊕): Create parallel reasoning streams

    Used for multi-chain reasoning where each branch explores different strategy
    """
    branches = []
    for _ in range(n_branches):
        # Deep copy context
        branch = HourglassContext()
        branch.past_tokens = context.past_tokens.copy()
        branch.semantic_history = context.semantic_history.copy()
        branch.chi = context.chi
        branch.state = context.state
        branches.append(branch)

    return branches


def join_streams(branches: List[HourglassContext],
                mode: str = "weighted") -> HourglassContext:
    """
    Join (⊗): Merge parallel streams

    Args:
        branches: List of parallel contexts
        mode: "weighted" (by max hazard) or "consensus" (majority vote)

    Returns:
        Merged context
    """
    if mode == "weighted":
        # Weight by maximum hazard in each branch
        weights = []
        for branch in branches:
            if len(branch.commitment_history) > 0:
                weights.append(max(branch.commitment_history))
            else:
                weights.append(0.0)

        # Select branch with highest weight
        best_idx = np.argmax(weights)
        return branches[best_idx]

    elif mode == "consensus":
        # Majority vote on token sequences
        # (Simplified: just pick longest sequence)
        longest_idx = np.argmax([len(b.past_tokens) for b in branches])
        return branches[longest_idx]

    else:
        raise ValueError(f"Unknown join mode: {mode}")


# ============================================================================
# Testing and Validation
# ============================================================================

def test_vbc_basic():
    """Basic test of VBC tick cycle"""
    print("Testing VBC basic functionality...")

    vbc = VariableBarrierController(h_star=0.4, top_k=5)

    # Simulate logits for vocabulary of 100 tokens
    logits = np.random.randn(100)
    logits[42] = 5.0  # Make token 42 most likely

    # Run through one complete tick cycle
    result = None
    for tick in range(8):  # 4 phases × 2 cycles
        result = vbc.process_logits(logits)
        if result is not None:
            print(f"✓ Committed to token {result.token_id} after {tick+1} ticks")
            print(f"  Hazard: {result.hazard:.3f}")
            print(f"  ε: {result.epsilon:.3f}, g: {result.g_phi:.3f}, ζ: {result.zeta:.3f}")
            print(f"  u: {result.u:.3f}, p: {result.p:.3f}")
            break

    if result is None:
        print("✗ No commit after 8 ticks (may need to lower h_star)")

    return result is not None


def test_vbc_chi_update():
    """Test χ criticality update"""
    print("\nTesting χ criticality update...")

    vbc = VariableBarrierController(h_star=0.3, top_k=10)

    # High-confidence logits (low entropy → high χ)
    logits_confident = np.zeros(100)
    logits_confident[0] = 10.0  # One dominant token

    # Run CAPTURE and BRIDGE to update χ
    vbc.process_logits(logits_confident)  # CAPTURE
    print(f"  After CAPTURE: {len(vbc.context.candidate_futures)} candidates")
    if len(vbc.context.candidate_futures) > 0:
        probs = [c.p for c in vbc.context.candidate_futures]
        print(f"  Probabilities: {probs[:3]}")

    vbc.process_logits(logits_confident)  # CLEAN
    print(f"  After CLEAN: {len(vbc.context.candidate_futures)} candidates")

    vbc.process_logits(logits_confident)  # BRIDGE
    print(f"  After BRIDGE: {len(vbc.context.candidate_futures)} candidates")

    if len(vbc.context.candidate_futures) > 0:
        epsilons = [c.epsilon for c in vbc.context.candidate_futures]
        print(f"  Epsilons: {epsilons[:3]}")  # Show first 3

    chi_confident = vbc.context.chi
    state_confident = vbc.context.state

    print(f"High-confidence: χ = {chi_confident:.3f}, state = {state_confident.value}")

    # Low-confidence logits (high entropy → low χ)
    vbc2 = VariableBarrierController(h_star=0.3, top_k=10)
    logits_uncertain = np.random.randn(100) * 0.5  # All similar

    vbc2.process_logits(logits_uncertain)  # CAPTURE
    vbc2.process_logits(logits_uncertain)  # CLEAN
    vbc2.process_logits(logits_uncertain)  # BRIDGE

    chi_uncertain = vbc2.context.chi
    state_uncertain = vbc2.context.state

    print(f"Low-confidence: χ = {chi_uncertain:.3f}, state = {state_uncertain.value}")

    # Validation: confident should have higher χ
    success = chi_confident > chi_uncertain
    print(f"{'✓' if success else '✗'} χ_confident > χ_uncertain: {success}")

    return success


def test_split_join():
    """Test parallel stream operations"""
    print("\nTesting split/join operations...")

    context = HourglassContext()
    context.past_tokens = [1, 2, 3]
    context.commitment_history = [0.6, 0.7, 0.8]

    # Split into 3 branches
    branches = split_stream(context, n_branches=3)
    print(f"✓ Split into {len(branches)} branches")

    # Simulate different evolution
    branches[0].commitment_history.append(0.9)
    branches[1].commitment_history.append(0.5)
    branches[2].commitment_history.append(0.7)

    # Join by weighted
    merged = join_streams(branches, mode="weighted")
    print(f"✓ Joined branches, max hazard = {max(merged.commitment_history):.2f}")

    # Should select branch 0 (highest final hazard)
    success = merged.commitment_history[-1] == 0.9
    print(f"{'✓' if success else '✗'} Selected highest-hazard branch: {success}")

    return success


if __name__ == "__main__":
    print("=" * 60)
    print("VBC Prototype Test Suite")
    print("=" * 60)

    results = []

    results.append(("Basic VBC", test_vbc_basic()))
    results.append(("Chi Update", test_vbc_chi_update()))
    results.append(("Split/Join", test_split_join()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for name, result in results if result)
    print(f"\nPassed {passed}/{total} tests ({100*passed/total:.0f}%)")
