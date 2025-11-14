"""
Δ-State: Core state object for Δ-compliant trading system

This replaces scattered state across modules with a unified,
auditable state structure that all components read/write.

Follows oracle specification for rigorous epistemics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np


class OperatingMode(Enum):
    """Operating mode with different evidence requirements."""
    RESEARCH = "research"      # E0-E2 only, no capital
    MICRO_LIVE = "micro_live"  # Tiny capital, live E3/E4, high logging
    PRODUCTION = "production"  # Full capital, all gates active


class LockStatus(Enum):
    """Lock validation status through E-gates."""
    DETECTED = "detected"      # Raw signal detected
    E0_PASSED = "e0_passed"    # Basic structure exists
    E1_PASSED = "e1_passed"    # Beats simple nulls
    E2_PASSED = "e2_passed"    # RG-stable
    E3_PASSED = "e3_passed"    # Live performance validated
    E4_PASSED = "e4_passed"    # Long-term robust
    ACTUALIZED = "actualized"  # PAD conditions met, ready to trade
    REJECTED = "rejected"      # Failed at some gate


@dataclass
class PhaseLock:
    """
    Phase-lock between two assets.

    Core Δ-Method properties:
    - Low-order wins: p, q small preferred (MDL penalty)
    - PAD: ε_cap, ε_stab, ζ determine if lock is actionable
    - E-audits: track validation level
    """
    # Identity
    pair: Tuple[str, str]
    p: int  # Order of first asset
    q: int  # Order of second asset

    # Coupling properties
    K: float  # Coupling strength
    Gamma_a: float  # Dissipation for asset A
    Gamma_b: float  # Dissipation for asset B
    Q_a: float  # Quality factor A (low dissipation)
    Q_b: float  # Quality factor B

    # PAD conditions (Potential → Actualized → Deployed)
    eps_cap: float  # Eligibility capacity (0 = closed, 1 = open)
    eps_stab: float  # Stability eligibility
    zeta: float  # Brittleness (concentration, overfit risk)

    # Evidence metrics
    delta_H_star: float = 0.0  # Harmony/evidence gain (replaces arbitrary Sharpe)
    status: LockStatus = LockStatus.DETECTED

    # E-gate results
    e0_passed: bool = False  # Structure exists
    e1_passed: bool = False  # Beats nulls
    e2_passed: bool = False  # RG-stable
    e3_passed: bool = False  # Live validated
    e4_passed: bool = False  # Long-term robust

    # Audit trail
    null_test_pvalues: Dict[str, float] = field(default_factory=dict)
    rg_test_results: Dict[str, Any] = field(default_factory=dict)
    live_performance: Dict[str, float] = field(default_factory=dict)

    # Detection metadata
    detected_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0  # Overall confidence (0-1)

    def mdl_penalty(self) -> float:
        """
        Low-Order Wins: penalize high-order locks.
        MDL penalty ~ log(p*q) or 1/(p*q)
        """
        return 1.0 / (self.p * self.q)

    def is_actionable(self) -> bool:
        """
        PAD conditions: lock is actionable if:
        - ε_cap > 0 (eligible to enter)
        - ε_stab > 0 (stable enough)
        - ζ ≤ ζ_max (not too brittle)
        - ΔH* > 0 (evidence gain positive)
        """
        ZETA_MAX = 0.7  # Brittleness threshold

        return (
            self.eps_cap > 0 and
            self.eps_stab > 0 and
            self.zeta <= ZETA_MAX and
            self.delta_H_star > 0 and
            self.status == LockStatus.ACTUALIZED
        )


@dataclass
class MarketRegime:
    """
    Market regime characterization.

    Note: χ and φ constants are HYPERPARAMETERS per regime,
    not universal constants (per oracle correction).
    """
    name: str  # OPTIMAL, ELEVATED, WARNING, CRISIS
    chi: float  # Current χ value
    chi_eq: float  # Equilibrium χ for this regime (regime-specific!)
    phi_prior: float  # φ prior for this regime (not universal!)

    # Regime properties
    volatility: float
    correlation: float
    liquidity: float

    # Confidence in regime classification
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditResult:
    """
    E-gate audit result.

    Each lock/strategy must pass E0→E1→E2→E3→E4 sequentially.
    """
    gate: str  # "E0", "E1", "E2", "E3", "E4"
    passed: bool
    metrics: Dict[str, float]
    null_comparison: Optional[Dict[str, Any]] = None
    rg_test: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class HazardItem:
    """
    Hazard item for trade selection (decoder analogy).

    Hazard law: h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·|∇T|

    Trades are "tokens" to be decoded; hazard determines which to select.
    """
    # Identity
    trade_id: str
    pair: Tuple[str, str]
    action: str  # "BUY", "SELL", "HOLD"

    # Hazard components
    epsilon: float  # Eligibility (risk limits, regime filters, liquidity)
    u: float  # Alignment between trade and χ/lock signals
    zeta: float  # Brittleness (concentration, leverage, overfit)
    g_e_phi: float  # Phase urge / timing factor
    kappa: float  # Gain coefficient

    # Computed hazard
    hazard: float  # h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·|∇T|

    # Expected impact
    delta_H_star: float  # Expected ΔH* from this trade
    cost_sigma: float  # Cost in coherence + money
    tur_ratio: float  # ΔH* / Σ (precision per entropy)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class Portfolio:
    """Portfolio state with risk metrics."""
    positions: Dict[str, float]  # ticker -> quantity
    cash: float
    total_value: float

    # Risk metrics
    max_drawdown: float
    time_underwater: int  # Days below peak
    current_drawdown: float

    # Performance
    daily_returns: List[float] = field(default_factory=list)
    sharpe: float = 0.0
    cagr: float = 0.0

    # FRACTAL LOW metrics (house score)
    crisis_survival_score: float = 0.0
    consistency_score: float = 0.0
    returns_score: float = 0.0
    house_score: float = 0.0


@dataclass
class DeltaState:
    """
    Unified Δ-State for the entire trading system.

    All modules (phase_lock_detector, chi_calculator, TUR, policies,
    LLM assistant) read/write this shared state.

    This replaces scattered state and makes auditing possible.
    """
    # Timestamp
    time: datetime = field(default_factory=datetime.now)

    # Operating mode (gates E-requirements)
    mode: OperatingMode = OperatingMode.RESEARCH

    # Raw market data
    markets: Dict[str, Any] = field(default_factory=dict)
    # Format: {
    #   "SPY": {"price": [...], "volume": [...], "timestamp": [...]},
    #   "AAPL": {...},
    #   ...
    # }

    # Detected phase locks (with E-audit status)
    locks: List[PhaseLock] = field(default_factory=list)

    # Market regimes (χ values are regime-specific hyperparameters)
    regimes: Dict[str, MarketRegime] = field(default_factory=dict)
    current_regime: Optional[str] = None

    # Portfolio state
    portfolio: Portfolio = field(default_factory=lambda: Portfolio(
        positions={},
        cash=100000.0,
        total_value=100000.0,
        max_drawdown=0.0,
        time_underwater=0,
        current_drawdown=0.0
    ))

    # Hazard items (trades ranked by decoder)
    hazards: List[HazardItem] = field(default_factory=list)

    # E-audit results for all locks/strategies
    audits: Dict[str, List[AuditResult]] = field(default_factory=dict)
    # Format: {"lock_id": [AuditResult(E0), AuditResult(E1), ...]}

    # Null test results
    null_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Mode promotion/demotion tracking
    mode_history: List[Tuple[datetime, OperatingMode, str]] = field(default_factory=list)

    # Logging / debug
    log: List[str] = field(default_factory=list)

    def add_log(self, message: str):
        """Add timestamped log entry."""
        self.log.append(f"[{self.time.isoformat()}] {message}")

    def get_actionable_locks(self) -> List[PhaseLock]:
        """Get locks that pass PAD conditions and are ready to trade."""
        return [lock for lock in self.locks if lock.is_actionable()]

    def get_locks_by_status(self, status: LockStatus) -> List[PhaseLock]:
        """Filter locks by validation status."""
        return [lock for lock in self.locks if lock.status == status]

    def update_portfolio_metrics(self):
        """Update portfolio risk/performance metrics."""
        if not self.portfolio.daily_returns:
            return

        returns = np.array(self.portfolio.daily_returns)

        # Sharpe
        if len(returns) > 1:
            self.portfolio.sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

        # CAGR
        if len(returns) > 252:
            total_return = (self.portfolio.total_value / 100000.0) - 1
            years = len(returns) / 252.0
            self.portfolio.cagr = (1 + total_return) ** (1 / years) - 1

        # Drawdown
        equity_curve = np.cumprod(1 + returns) * 100000.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        self.portfolio.current_drawdown = drawdown[-1]
        self.portfolio.max_drawdown = np.min(drawdown)

        # Time underwater
        self.portfolio.time_underwater = np.sum(drawdown < -0.01)

        # FRACTAL LOW house score
        # 50% crisis survival + 25% consistency + 25% returns
        crisis_survival = (1.0 - min(1.0, abs(self.portfolio.max_drawdown) / 0.55)) * 50

        # Consistency: win rate × 15 + profit factor × 2 + (1/std) × 8
        # (simplified here, full version needs trade-level data)
        consistency = min(np.std(returns), 0.1) * 100  # Inverse volatility

        # Returns: relative to SPY benchmark
        returns_score = min(self.portfolio.cagr / 0.078, 2.0) * 15

        self.portfolio.crisis_survival_score = crisis_survival
        self.portfolio.consistency_score = consistency
        self.portfolio.returns_score = returns_score
        self.portfolio.house_score = crisis_survival + consistency + returns_score


# Factory functions

def create_research_state() -> DeltaState:
    """Create state in RESEARCH mode (E0-E2 only, no capital)."""
    state = DeltaState(mode=OperatingMode.RESEARCH)
    state.portfolio.cash = 0.0  # No capital in research
    state.add_log("Initialized in RESEARCH mode (E0-E2 only)")
    return state


def create_micro_live_state(capital: float = 1000.0) -> DeltaState:
    """Create state in MICRO_LIVE mode (tiny capital, live E3/E4)."""
    state = DeltaState(mode=OperatingMode.MICRO_LIVE)
    state.portfolio.cash = capital
    state.portfolio.total_value = capital
    state.add_log(f"Initialized in MICRO_LIVE mode with ${capital:.2f}")
    return state


def create_production_state(capital: float) -> DeltaState:
    """Create state in PRODUCTION mode (full capital, all gates)."""
    state = DeltaState(mode=OperatingMode.PRODUCTION)
    state.portfolio.cash = capital
    state.portfolio.total_value = capital
    state.add_log(f"Initialized in PRODUCTION mode with ${capital:,.2f}")
    return state
