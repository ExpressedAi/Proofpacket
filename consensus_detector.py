"""
Redundancy Consensus Detector for Δ-Trading System
Layer 1: Opportunity Detection with Multi-Signal Redundancy

This module implements the consensus detection mechanism where we wait for
MULTIPLE independent signals to align before entering a trade. This reduces
false positives and increases win rate.

Key Concept:
    R = Σ signal_strength_i (fractional sum, not binary count)
    R* = threshold for consensus (default 3.5 out of 5.0)
    dR/dt = rate of consensus formation (positive = building)

    ENTER when: R >= R* AND dR/dt > 0
    EXIT when: R < R* or dR/dt < 0 for sustained period
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.signal import hilbert
from scipy.stats import zscore


def sigmoid(x: float) -> float:
    """Smooth activation function for continuous signal strength."""
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class MarketState:
    """State vector for a trading pair at time t."""
    timestamp: float
    pair: Tuple[str, str]  # e.g., ("AAPL", "MSFT")

    # Phase-lock metrics
    phase_diff: float  # |φ_a - φ_b|
    K: float  # Coupling strength
    eps: float  # Eligibility = [2πK - (Γ_a + Γ_b)]₊
    zeta: float  # Brittleness

    # Hazard rate
    h: float  # h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·|∇T|

    # Criticality
    chi: float  # χ = flux/dissipation

    # E3 validation
    e3_passed: bool  # Causal direction test
    e3_score: float  # Continuous score

    # Previous redundancy (for dR/dt)
    R_prev: float = 0.0


@dataclass
class ConsensusSignal:
    """Result of consensus detection."""
    enter: bool  # Should we enter?
    exit: bool  # Should we exit?
    strength: float  # R / 5.0 (normalized)
    urgency: float  # dR/dt (rate of formation)
    breakdown: Dict[str, float]  # Individual signal strengths


class ConsensusDetector:
    """
    Detects when multiple independent signals align (redundancy threshold).

    Five independent signals:
        1. Hazard: h(t) > threshold (commitment imminent)
        2. Eligibility: ε > threshold (window open)
        3. Causal: E3 test passed (causal link confirmed)
        4. Stability: χ < threshold (not in crisis)
        5. Robustness: ζ < threshold (not brittle)
    """

    def __init__(
        self,
        R_star: float = 3.5,  # Consensus threshold (out of 5.0)
        h_threshold: float = 0.6,  # Hazard threshold
        eps_threshold: float = 0.2,  # Eligibility threshold
        chi_threshold: float = 1.0,  # Stability threshold
        zeta_threshold: float = 0.7,  # Robustness threshold
        dR_dt_window: int = 5,  # Window for computing dR/dt
        exit_lag: int = 3,  # Require R < R* for N ticks before exit
    ):
        self.R_star = R_star
        self.h_threshold = h_threshold
        self.eps_threshold = eps_threshold
        self.chi_threshold = chi_threshold
        self.zeta_threshold = zeta_threshold
        self.dR_dt_window = dR_dt_window
        self.exit_lag = exit_lag

        # History for dR/dt computation
        self.R_history: Dict[Tuple[str, str], List[float]] = {}

        # Exit lag counter
        self.below_threshold_count: Dict[Tuple[str, str], int] = {}

    def compute_signal_strengths(self, state: MarketState) -> Dict[str, float]:
        """
        Compute continuous signal strengths [0, 1] for each of 5 signals.

        Returns:
            Dictionary of signal_name -> strength
        """
        signals = {}

        # 1. Hazard signal: h(t) approaching commitment
        # High when h > h_threshold
        h_score = (state.h - self.h_threshold) / 0.1  # Normalized
        signals['hazard'] = sigmoid(h_score)

        # 2. Eligibility signal: window open for locks
        # High when ε > eps_threshold
        eps_score = state.eps / self.eps_threshold - 1.0
        signals['eligibility'] = sigmoid(eps_score)

        # 3. Causal signal: E3 validation passed
        # Binary + continuous E3 score
        if state.e3_passed:
            signals['causal'] = min(1.0, 0.5 + state.e3_score)
        else:
            signals['causal'] = 0.0

        # 4. Stability signal: NOT in crisis
        # High when χ < chi_threshold
        chi_score = (self.chi_threshold - state.chi) / 0.3
        signals['stability'] = sigmoid(chi_score)

        # 5. Robustness signal: NOT brittle
        # High when ζ < zeta_threshold
        zeta_score = (self.zeta_threshold - state.zeta) / 0.2
        signals['robustness'] = sigmoid(zeta_score)

        return signals

    def compute_redundancy(self, signals: Dict[str, float]) -> float:
        """
        Compute redundancy R = sum of signal strengths.

        R ∈ [0, 5.0] where:
            R = 0: No signals
            R = 2.5: Half signals
            R = 5.0: All signals maxed
        """
        return sum(signals.values())

    def compute_dR_dt(self, pair: Tuple[str, str], R: float) -> float:
        """
        Compute rate of consensus formation: dR/dt.

        Positive dR/dt = consensus building (good time to enter)
        Negative dR/dt = consensus breaking (consider exit)
        """
        if pair not in self.R_history:
            self.R_history[pair] = []

        self.R_history[pair].append(R)

        # Keep only recent window
        if len(self.R_history[pair]) > self.dR_dt_window:
            self.R_history[pair] = self.R_history[pair][-self.dR_dt_window:]

        # Need at least 2 points for derivative
        if len(self.R_history[pair]) < 2:
            return 0.0

        # Linear regression slope
        history = np.array(self.R_history[pair])
        x = np.arange(len(history))

        # dR/dt ≈ slope of linear fit
        slope = np.polyfit(x, history, deg=1)[0]

        return slope

    def should_enter(
        self,
        state: MarketState,
        signals: Dict[str, float],
        R: float,
        dR_dt: float
    ) -> bool:
        """
        Determine if we should enter a position.

        Entry conditions:
            1. R >= R* (consensus reached)
            2. dR/dt > 0 (consensus still forming, not decaying)
            3. Eligibility > minimum (ε > 0.1, not closed)
        """
        consensus_reached = R >= self.R_star
        consensus_building = dR_dt > 0
        window_open = state.eps > 0.1  # Minimum eligibility

        return consensus_reached and consensus_building and window_open

    def should_exit(
        self,
        pair: Tuple[str, str],
        R: float,
        dR_dt: float
    ) -> bool:
        """
        Determine if we should exit a position.

        Exit conditions:
            1. R < R* for exit_lag consecutive ticks
            2. OR dR/dt < -0.5 (rapid consensus breakdown)
        """
        if pair not in self.below_threshold_count:
            self.below_threshold_count[pair] = 0

        # Check if below threshold
        if R < self.R_star:
            self.below_threshold_count[pair] += 1
        else:
            self.below_threshold_count[pair] = 0

        # Exit if sustained weakness OR rapid breakdown
        sustained_weakness = self.below_threshold_count[pair] >= self.exit_lag
        rapid_breakdown = dR_dt < -0.5

        return sustained_weakness or rapid_breakdown

    def detect(self, state: MarketState) -> ConsensusSignal:
        """
        Main entry point: detect consensus for a given market state.

        Returns:
            ConsensusSignal with entry/exit decisions and metrics
        """
        # Compute all signal strengths
        signals = self.compute_signal_strengths(state)

        # Compute redundancy
        R = self.compute_redundancy(signals)

        # Compute rate of formation
        dR_dt = self.compute_dR_dt(state.pair, R)

        # Entry/exit decisions
        enter = self.should_enter(state, signals, R, dR_dt)
        exit_signal = self.should_exit(state.pair, R, dR_dt)

        # Normalized strength
        strength = R / 5.0

        return ConsensusSignal(
            enter=enter,
            exit=exit_signal,
            strength=strength,
            urgency=dR_dt,
            breakdown=signals
        )

    def reset_history(self, pair: Tuple[str, str]):
        """Clear history for a pair (e.g., after exit)."""
        if pair in self.R_history:
            del self.R_history[pair]
        if pair in self.below_threshold_count:
            del self.below_threshold_count[pair]


# ============================================================================
# Phase-Lock Detection Utilities (from fibonacci_triad_quick_analysis.py)
# ============================================================================

def compute_phase_difference(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase difference between two time series.

    Uses Hilbert transform to extract analytic signal.
    """
    # Normalize
    a = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-10)
    b = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-10)

    # Hilbert transform
    analytic_a = hilbert(a)
    analytic_b = hilbert(b)

    # Extract phase
    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)

    # Phase difference
    delta_phi = np.abs(phase_a - phase_b)

    # Wrap to [0, π]
    delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)

    return delta_phi


def compute_coupling_strength(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 20
) -> float:
    """
    Compute coupling strength K for a pair.

    K = <cos(Δφ)> where Δφ = φ_a - φ_b

    Returns:
        K ∈ [-1, 1] where 1 = perfect lock, -1 = perfect anti-lock
    """
    delta_phi = compute_phase_difference(series_a, series_b)

    # Rolling window average
    if len(delta_phi) < window:
        window = len(delta_phi)

    recent_delta_phi = delta_phi[-window:]

    # K = <cos(Δφ)>
    K = np.mean(np.cos(recent_delta_phi))

    return K


def compute_eligibility(
    K: float,
    gamma_a: float,
    gamma_b: float,
    scale: float = 1.0
) -> float:
    """
    Compute eligibility ε = [2πK - (Γ_a + Γ_b)]₊

    Args:
        K: Coupling strength
        gamma_a: Decay rate for series A
        gamma_b: Decay rate for series B
        scale: Scale factor (default 1.0)

    Returns:
        ε >= 0 (window size for lock formation)
    """
    eps = 2 * np.pi * K - (gamma_a + gamma_b)
    return max(0.0, eps * scale)


def compute_brittleness(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 20
) -> float:
    """
    Compute brittleness ζ = volatility of phase difference.

    High ζ = brittle (phase wanders)
    Low ζ = robust (phase stable)
    """
    delta_phi = compute_phase_difference(series_a, series_b)

    if len(delta_phi) < window:
        window = len(delta_phi)

    recent_delta_phi = delta_phi[-window:]

    # Brittleness = standard deviation of phase
    zeta = np.std(recent_delta_phi)

    return zeta


def compute_hazard_rate(
    K: float,
    eps: float,
    e_phi: float,
    zeta: float,
    zeta_star: float = 0.7,
    grad_T: float = 1.0,
    kappa: float = 1.0
) -> float:
    """
    Compute hazard rate h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·|∇T|

    This predicts WHEN a phase-lock will commit.

    Args:
        K: Coupling strength
        eps: Eligibility
        e_phi: Phase error metric
        zeta: Brittleness
        zeta_star: Brittleness threshold
        grad_T: Temperature gradient (default 1.0)
        kappa: Scaling constant

    Returns:
        h >= 0 (hazard rate, higher = commitment imminent)
    """
    # g(e_φ) ≈ exp(-e_φ) (Gaussian gate)
    g = np.exp(-e_phi)

    # Brittleness factor
    brittleness_factor = max(0.0, 1.0 - zeta / zeta_star)

    h = kappa * eps * g * brittleness_factor * grad_T

    return max(0.0, h)


def compute_chi(
    series: np.ndarray,
    flux_window: int = 5,
    dissipation_window: int = 20
) -> float:
    """
    Compute χ-criticality: χ = flux / dissipation

    High χ (> 0.618) = crisis mode
    Low χ (< 0.618) = stable

    Args:
        series: Price series
        flux_window: Window for flux (short-term volatility)
        dissipation_window: Window for dissipation (long-term mean reversion)

    Returns:
        χ >= 0
    """
    if len(series) < dissipation_window:
        return 0.0

    # Flux = short-term volatility
    recent = series[-flux_window:]
    flux = np.std(recent) / (np.mean(np.abs(recent)) + 1e-10)

    # Dissipation = mean reversion rate
    long_term = series[-dissipation_window:]
    returns = np.diff(long_term) / (long_term[:-1] + 1e-10)
    dissipation = 1.0 / (np.std(returns) + 1e-10)

    chi = flux / (dissipation + 1e-10)

    return chi


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Consensus Detector - Example Usage")
    print("=" * 60)

    # Create detector
    detector = ConsensusDetector(
        R_star=3.5,
        h_threshold=0.6,
        eps_threshold=0.2
    )

    # Example market state (simulated)
    state = MarketState(
        timestamp=1.0,
        pair=("AAPL", "MSFT"),
        phase_diff=0.15,  # Small phase difference (locked)
        K=0.85,  # Strong coupling
        eps=0.35,  # Eligibility window open
        zeta=0.45,  # Low brittleness (robust)
        h=0.72,  # High hazard (commitment imminent)
        chi=0.42,  # Low criticality (stable market)
        e3_passed=True,
        e3_score=0.65,
        R_prev=0.0
    )

    # Detect consensus
    signal = detector.detect(state)

    print(f"\nMarket State:")
    print(f"  Pair: {state.pair}")
    print(f"  Coupling K: {state.K:.3f}")
    print(f"  Eligibility ε: {state.eps:.3f}")
    print(f"  Hazard h: {state.h:.3f}")
    print(f"  Brittleness ζ: {state.zeta:.3f}")
    print(f"  Criticality χ: {state.chi:.3f}")

    print(f"\nSignal Breakdown:")
    for signal_name, strength in signal.breakdown.items():
        bar = "█" * int(strength * 20)
        print(f"  {signal_name:12s}: {strength:.3f} {bar}")

    print(f"\nConsensus Metrics:")
    print(f"  Redundancy R: {signal.strength * 5.0:.2f} / 5.0")
    print(f"  Threshold R*: {detector.R_star:.2f}")
    print(f"  Formation rate dR/dt: {signal.urgency:.3f}")

    print(f"\nDecision:")
    print(f"  ENTER: {signal.enter}")
    print(f"  EXIT: {signal.exit}")
    print(f"  Strength: {signal.strength:.1%}")

    print("\n" + "=" * 60)
    print("✓ Day 1 implementation complete")
    print("  Next: Day 2 - χ crash prediction backtest")
