"""
Δ-State: Canonical State Object (Oracle-Compliant)

Single source of truth for entire trading system.
All modules read/write ONLY through this state.

No hidden globals. No ad-hoc state in modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np


class OperatingMode(Enum):
    """Operating mode with different evidence requirements."""
    RESEARCH = "research"      # E0-E2 only, no capital
    MICRO_LIVE = "micro_live"  # Tiny capital, live E3/E4, high logging
    PRODUCTION = "production"  # Full stack, capital scaled by evidence


class RegimeLabel(Enum):
    """Market regime classification."""
    NORMAL = "normal"
    CRISIS = "crisis"
    TRANSITION = "transition"


class EStatus(Enum):
    """E-gate status."""
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"


# ============================================================================
# Market Data
# ============================================================================

@dataclass
class MarketSeries:
    """Raw and derived time series for a single asset."""
    symbol: str
    dates: np.ndarray          # shape (T,)
    prices: np.ndarray         # shape (T,)
    volumes: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None  # log or simple (be consistent)
    features: Dict[str, np.ndarray] = field(default_factory=dict)  # vol, factors, etc.

    def __post_init__(self):
        """Compute returns if not provided."""
        if self.returns is None and len(self.prices) > 1:
            self.returns = np.diff(np.log(self.prices))


# ============================================================================
# Lock State (Phase-Lock Detection)
# ============================================================================

@dataclass
class LockState:
    """
    One detected low-order lock between assets or factors.

    Represents p:q phase-lock with full PAD tracking.
    """
    # Identity
    id: str                    # unique ID (e.g., "AAPL-MSFT-2:3")
    a: str                     # symbol or factor name
    b: str
    p: int                     # ratio p:q
    q: int

    # Coupling properties
    K: float                   # coupling strength
    Gamma_a: float             # damping rate A
    Gamma_b: float             # damping rate B
    Q_a: float                 # quality factor A = f/Gamma
    Q_b: float                 # quality factor B

    # PAD conditions
    eps_cap: float             # capture eligibility (0-1)
    eps_stab: float            # stability window (0-1)
    zeta: float                # brittleness (0-1, higher = more brittle)

    # Metadata
    order: int = field(init=False)  # p+q (computed in __post_init__)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Evidence
    evidence_score: float = 0.0  # Aggregate ΔH*
    e_level_passed: int = 0      # Max E-level passed (0-4)

    # Audit results
    e0_status: EStatus = EStatus.PENDING
    e1_status: EStatus = EStatus.PENDING
    e2_status: EStatus = EStatus.PENDING
    e3_status: EStatus = EStatus.PENDING
    e4_status: EStatus = EStatus.PENDING

    def __post_init__(self):
        """Compute order from p, q."""
        self.order = self.p + self.q

    def mdl_penalty(self) -> float:
        """
        Low-Order Wins: weight for low-order locks.

        Returns higher values for low-order locks (p, q small).
        This is a "priority weight", not a "penalty to subtract".
        """
        return 1.0 / (self.p * self.q)

    def is_potential(self) -> bool:
        """P: Does lock meet basic potential criteria?"""
        return (
            self.order <= 7 and  # Low-order
            abs(self.K) > 0.1 and  # Non-trivial coupling
            self.e0_status == EStatus.PASS
        )

    def is_actualized(self) -> bool:
        """A: Does lock pass actualization conditions?"""
        return (
            self.is_potential() and
            self.eps_cap > 0.3 and
            self.eps_stab > 0.3 and
            self.zeta <= 0.7 and
            self.e1_status == EStatus.PASS and
            self.e2_status == EStatus.PASS
        )

    def is_deployable(self) -> bool:
        """D: Can lock be traded?"""
        return (
            self.is_actualized() and
            self.e_level_passed >= 3 and
            self.evidence_score > 0
        )


# ============================================================================
# Regime State
# ============================================================================

@dataclass
class RegimeState:
    """Global and per-asset regime labels + χ metrics."""
    timestamp: datetime
    global_regime: RegimeLabel
    chi_global: float          # global χ (e.g., from vol/lock density)
    chi_trend: float           # smoothed χ
    asset_regimes: Dict[str, RegimeLabel] = field(default_factory=dict)
    chi_assets: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Portfolio State
# ============================================================================

@dataclass
class Position:
    """Single position in portfolio."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    last_update: datetime
    pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        """Update unrealized P&L."""
        self.pnl = (current_price - self.entry_price) * self.quantity
        self.last_update = datetime.utcnow()


@dataclass
class PortfolioState:
    """Portfolio state with risk metrics."""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    def total_value(self) -> float:
        """Compute total portfolio value."""
        position_value = sum([pos.pnl for pos in self.positions.values()])
        return self.cash + position_value

    def update_exposures(self):
        """Update gross/net exposure."""
        gross = sum([abs(pos.quantity * pos.entry_price) for pos in self.positions.values()])
        net = sum([pos.quantity * pos.entry_price for pos in self.positions.values()])
        self.gross_exposure = gross
        self.net_exposure = net


# ============================================================================
# Hazard Item (Decoder Analogy)
# ============================================================================

@dataclass
class HazardItem:
    """
    One hazard candidate (for lock, trade, or strategy).

    Hazard law: h = κ·ε·g(e_φ)·(1-ζ/ζ*)·u·p
    """
    # Identity
    id: str                    # e.g. "lock:EURUSD-1:1" or "trade:AAPL-long"
    category: str              # "lock", "trade", "strategy"

    # Hazard components
    epsilon: float             # eligibility (risk limits, liquidity)
    g_phi: float               # timing / phase urge
    zeta: float                # brittleness (concentration, overfit)
    u: float                   # alignment with objective
    p: float                   # prior probability of success
    kappa: float               # gain coefficient

    # Computed
    hazard: float              # h = κ·ε·g·(1-ζ/ζ*)·u·p

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def compute_hazard(
        kappa: float,
        epsilon: float,
        g_phi: float,
        zeta: float,
        u: float,
        p: float,
        zeta_star: float = 1.0
    ) -> float:
        """Compute hazard using canonical formula."""
        brittleness_factor = max(0, 1 - zeta / zeta_star)
        return kappa * epsilon * g_phi * brittleness_factor * u * p


# ============================================================================
# Audit Stats (E-Gates)
# ============================================================================

@dataclass
class AuditStats:
    """
    E-audit status for one edge/strategy/lock.
    Each level: pending/pass/fail.
    """
    id: str
    E0: EStatus = EStatus.PENDING
    E1: EStatus = EStatus.PENDING
    E2: EStatus = EStatus.PENDING
    E3: EStatus = EStatus.PENDING
    E4: EStatus = EStatus.PENDING
    last_updated: datetime = field(default_factory=datetime.utcnow)
    notes: Dict[str, Any] = field(default_factory=dict)

    def max_level_passed(self) -> int:
        """Return max E-level passed (0-4)."""
        levels = [self.E0, self.E1, self.E2, self.E3, self.E4]
        for i in range(4, -1, -1):
            if levels[i] == EStatus.PASS:
                return i
        return 0

    def all_passed_up_to(self, level: int) -> bool:
        """Check if all levels up to `level` are passed."""
        levels = [self.E0, self.E1, self.E2, self.E3, self.E4]
        return all(levels[i] == EStatus.PASS for i in range(level + 1))


# ============================================================================
# Strategy State (Per-Strategy Evidence)
# ============================================================================

@dataclass
class StrategyState:
    """
    Per-strategy evidence and mode.

    Strategy examples: 'chi_crash', 'consensus', 'fraud_filter', etc.
    """
    name: str
    operating_mode: OperatingMode
    audit: AuditStats

    # Evidence
    evidence_score: float = 0.0        # Aggregate ΔH*
    risk_budget_fraction: float = 0.0  # 0-1 of total capital
    live_capital: float = 0.0

    # Mode history
    last_promotion: Optional[datetime] = None
    last_demotion: Optional[datetime] = None
    mode_history: List[Tuple[datetime, OperatingMode, str]] = field(default_factory=list)

    def log_mode_change(self, new_mode: OperatingMode, reason: str):
        """Record mode change."""
        timestamp = datetime.utcnow()
        self.mode_history.append((timestamp, new_mode, reason))

        if new_mode.value > self.operating_mode.value:
            self.last_promotion = timestamp
        else:
            self.last_demotion = timestamp

        self.operating_mode = new_mode


# ============================================================================
# Canonical Δ-State
# ============================================================================

@dataclass
class DeltaState:
    """
    Canonical state shared across the entire Δ trading system.

    All modules read/write ONLY through this state.
    No hidden globals. No ad-hoc state.
    """
    # Core metadata
    timestamp: datetime
    operating_mode: OperatingMode

    # Market data
    markets: Dict[str, MarketSeries] = field(default_factory=dict)

    # Locks (phase-locks detected)
    locks: Dict[str, LockState] = field(default_factory=dict)

    # Regime
    regime: Optional[RegimeState] = None

    # Portfolio
    portfolio: Optional[PortfolioState] = None

    # Hazards (decoder candidates)
    hazards: List[HazardItem] = field(default_factory=list)

    # Audits (E-gate status per entity)
    audits: Dict[str, AuditStats] = field(default_factory=dict)

    # Strategies (per-strategy evidence + mode)
    strategies: Dict[str, StrategyState] = field(default_factory=dict)

    # Metadata / logging
    meta: Dict[str, Any] = field(default_factory=dict)
    log: List[Tuple[datetime, str]] = field(default_factory=list)

    def add_log(self, message: str):
        """Add timestamped log entry."""
        self.log.append((self.timestamp, message))

    def get_deployable_locks(self) -> List[LockState]:
        """Get locks that can be traded (PAD complete)."""
        return [lock for lock in self.locks.values() if lock.is_deployable()]

    def get_actualized_locks(self) -> List[LockState]:
        """Get locks that passed actualization."""
        return [lock for lock in self.locks.values() if lock.is_actualized()]

    def update_portfolio_value(self):
        """Update portfolio total value and equity curve."""
        if self.portfolio:
            total = self.portfolio.total_value()
            self.portfolio.equity_curve.append((self.timestamp, total))

            # Update drawdown
            if self.portfolio.equity_curve:
                values = [v for _, v in self.portfolio.equity_curve]
                peak = max(values)
                current = values[-1]
                drawdown = (current - peak) / peak if peak > 0 else 0
                self.portfolio.current_drawdown = drawdown
                self.portfolio.max_drawdown = min(self.portfolio.max_drawdown, drawdown)


# ============================================================================
# Factory Functions
# ============================================================================

def create_research_state() -> DeltaState:
    """Create state in RESEARCH mode (E0-E2 only, no capital)."""
    state = DeltaState(
        timestamp=datetime.utcnow(),
        operating_mode=OperatingMode.RESEARCH,
        portfolio=PortfolioState(
            timestamp=datetime.utcnow(),
            cash=0.0  # No capital in research
        )
    )
    state.add_log("Initialized in RESEARCH mode (E0-E2 only)")
    return state


def create_micro_live_state(capital: float = 1000.0) -> DeltaState:
    """Create state in MICRO_LIVE mode (tiny capital, live E3/E4)."""
    state = DeltaState(
        timestamp=datetime.utcnow(),
        operating_mode=OperatingMode.MICRO_LIVE,
        portfolio=PortfolioState(
            timestamp=datetime.utcnow(),
            cash=capital
        )
    )
    state.add_log(f"Initialized in MICRO_LIVE mode with ${capital:.2f}")
    return state


def create_production_state(capital: float = 100000.0) -> DeltaState:
    """Create state in PRODUCTION mode (full capital, all gates)."""
    state = DeltaState(
        timestamp=datetime.utcnow(),
        operating_mode=OperatingMode.PRODUCTION,
        portfolio=PortfolioState(
            timestamp=datetime.utcnow(),
            cash=capital
        )
    )
    state.add_log(f"Initialized in PRODUCTION mode with ${capital:,.2f}")
    return state


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Δ-STATE V2: Oracle-Compliant Canonical State")
    print("="*70)

    # Create research state
    state = create_research_state()

    # Add market data
    dates = np.arange(100)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    state.markets["AAPL"] = MarketSeries(
        symbol="AAPL",
        dates=dates,
        prices=prices
    )

    # Add a lock
    lock = LockState(
        id="AAPL-MSFT-2:3",
        a="AAPL",
        b="MSFT",
        p=2,
        q=3,
        K=0.65,
        Gamma_a=0.1,
        Gamma_b=0.1,
        Q_a=10,
        Q_b=10,
        eps_cap=0.8,
        eps_stab=0.7,
        zeta=0.3
    )
    lock.e0_status = EStatus.PASS
    lock.e1_status = EStatus.PASS
    lock.e2_status = EStatus.PASS
    lock.e_level_passed = 2

    state.locks[lock.id] = lock

    # Add strategy
    audit = AuditStats(id="consensus", E0=EStatus.PASS, E1=EStatus.PASS)
    strategy = StrategyState(
        name="consensus",
        operating_mode=OperatingMode.RESEARCH,
        audit=audit,
        evidence_score=5.0
    )
    state.strategies["consensus"] = strategy

    # Add hazard item
    hazard = HazardItem(
        id="trade:AAPL-long",
        category="trade",
        epsilon=0.8,
        g_phi=0.7,
        zeta=0.3,
        u=0.9,
        p=0.6,
        kappa=1.0,
        hazard=HazardItem.compute_hazard(1.0, 0.8, 0.7, 0.3, 0.9, 0.6)
    )
    state.hazards.append(hazard)

    # Print state
    print(f"\nState timestamp: {state.timestamp}")
    print(f"Operating mode: {state.operating_mode.value}")
    print(f"Markets: {list(state.markets.keys())}")
    print(f"Locks: {len(state.locks)}")
    print(f"Strategies: {list(state.strategies.keys())}")
    print(f"Hazards: {len(state.hazards)}")

    # Check lock PAD
    print(f"\nLock {lock.id}:")
    print(f"  Potential: {lock.is_potential()}")
    print(f"  Actualized: {lock.is_actualized()}")
    print(f"  Deployable: {lock.is_deployable()}")
    print(f"  E-level passed: {lock.e_level_passed}")

    print("\n" + "="*70)
