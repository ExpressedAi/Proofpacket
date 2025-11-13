"""
VBC Trade Decoder: Variable Barrier Controller for Trade Selection

Cross-ontological decoder analogy:
- LLM picks tokens by maximizing h(token | context)
- Trading picks trades by maximizing h(trade | market state)

Hazard Law (Canonical):
    h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·u·p

Where:
- κ = gain coefficient (expected profit / volatility)
- ε = eligibility (risk limits, liquidity, regime filters)
- g(e_φ) = phase urge / timing factor
- ζ = brittleness (concentration, leverage, overfit)
- u = alignment (trade direction vs signals)
- p = prior probability of success

The decoder selects trades that maximize hazard subject to constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import numpy as np

from delta_state_v2 import (
    DeltaState, HazardItem, LockState, PortfolioState,
    RegimeState, RegimeLabel
)


# ============================================================================
# Trade Candidate (Token Analogy)
# ============================================================================

@dataclass
class TradeCandidate:
    """
    Single trade candidate for decoder.

    Analogous to TokenCandidate in LLM decoding.
    """
    # Identity
    trade_id: str
    lock_id: str  # Which lock generated this trade
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"

    # Position sizing
    quantity: float
    entry_price: float

    # Hazard components (computed)
    kappa: float = 0.0  # Gain coefficient
    epsilon: float = 0.0  # Eligibility
    g_phi: float = 0.0  # Phase urge
    zeta: float = 0.0  # Brittleness
    u: float = 0.0  # Alignment
    p: float = 0.0  # Prior success probability

    # Computed hazard
    hazard: float = 0.0

    # Expected impact
    expected_pnl: float = 0.0
    expected_delta_h: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""


# ============================================================================
# Hazard Calculation
# ============================================================================

def compute_kappa(
    expected_return: float,
    volatility: float
) -> float:
    """
    Gain coefficient κ = E[return] / σ

    Normalized Sharpe-like metric.
    """
    return expected_return / (volatility + 1e-6)


def compute_epsilon(
    portfolio: PortfolioState,
    regime: RegimeState,
    position_size: float,
    max_position_fraction: float = 0.1
) -> float:
    """
    Eligibility ε ∈ [0, 1]

    Factors:
    - Risk limits: Position size vs total capital
    - Regime filters: Don't trade in CRISIS
    - Liquidity: Can we execute this size?

    Returns 0 if ineligible, 1 if fully eligible.
    """
    # Check position size limit
    position_value = abs(position_size)
    max_position = portfolio.total_value() * max_position_fraction

    if position_value > max_position:
        size_eligibility = max_position / position_value
    else:
        size_eligibility = 1.0

    # Check regime
    if regime.global_regime == RegimeLabel.CRISIS:
        regime_eligibility = 0.0  # Block all trades in crisis
    elif regime.global_regime == RegimeLabel.TRANSITION:
        regime_eligibility = 0.5  # Reduced trading in transition
    else:
        regime_eligibility = 1.0  # Full trading in normal

    # Combined eligibility
    epsilon = size_eligibility * regime_eligibility

    return np.clip(epsilon, 0.0, 1.0)


def compute_g_phi(
    lock: LockState,
    current_phase_diff: float
) -> float:
    """
    Phase urge g(e_φ)

    Measures how urgent it is to act based on phase alignment.

    When phase difference is near optimal entry point, g is high.
    """
    # Optimal phase for p:q lock
    # Simplified: g = |cos(phase_diff)|
    g = abs(np.cos(current_phase_diff))

    return g


def compute_zeta_trade(
    portfolio: PortfolioState,
    symbol: str,
    position_size: float
) -> float:
    """
    Brittleness ζ ∈ [0, 1] for this trade.

    Factors:
    - Concentration: What fraction of portfolio is in this symbol?
    - Leverage: Are we overleveraged?
    - Existing exposure: Do we already have a position?

    Higher ζ = more brittle (concentrated, risky)
    """
    # Concentration
    existing_position = portfolio.positions.get(symbol, None)
    existing_value = (
        abs(existing_position.quantity * existing_position.entry_price)
        if existing_position else 0.0
    )

    new_value = abs(position_size)
    total_value = portfolio.total_value()

    concentration = (existing_value + new_value) / total_value

    # Leverage (gross exposure / total value)
    leverage_factor = portfolio.gross_exposure / total_value if total_value > 0 else 0

    # Combined brittleness
    zeta = 0.6 * concentration + 0.4 * min(leverage_factor, 1.0)

    return np.clip(zeta, 0.0, 1.0)


def compute_u_alignment(
    trade_direction: str,
    signal_strengths: Dict[str, float]
) -> float:
    """
    Alignment u ∈ [-1, 1]

    Measures how well trade aligns with all signals.

    u = average(signal_strengths weighted by reliability)
    """
    if not signal_strengths:
        return 0.0

    # Simple average for now
    # In production, would weight by signal reliability
    alignments = []

    for signal_name, strength in signal_strengths.items():
        # Assume strength > 0 means bullish, < 0 means bearish
        if trade_direction == "BUY":
            alignments.append(strength)
        elif trade_direction == "SELL":
            alignments.append(-strength)
        else:  # HOLD
            alignments.append(0.0)

    u = np.mean(alignments)

    return np.clip(u, -1.0, 1.0)


def compute_p_prior(
    lock: LockState,
    historical_win_rate: Optional[float] = None
) -> float:
    """
    Prior probability of success p ∈ [0, 1]

    Based on:
    - Lock's historical performance
    - Evidence score (ΔH*)
    - E-level passed

    Returns base rate if no history available.
    """
    if historical_win_rate is not None:
        p_historical = historical_win_rate
    else:
        p_historical = 0.5  # Base rate

    # Adjust by evidence score
    if lock.evidence_score > 0:
        # Higher ΔH* → higher success probability
        p_evidence = 0.5 + 0.3 * np.tanh(lock.evidence_score)
    else:
        p_evidence = 0.5

    # Adjust by E-level
    e_bonus = lock.e_level_passed * 0.05  # +5% per E-level

    p = p_historical * 0.5 + p_evidence * 0.5 + e_bonus

    return np.clip(p, 0.0, 1.0)


def compute_hazard_canonical(
    kappa: float,
    epsilon: float,
    g_phi: float,
    zeta: float,
    u: float,
    p: float,
    zeta_star: float = 1.0
) -> float:
    """
    Canonical hazard law:

    h = κ·ε·g·(1-ζ/ζ*)·u·p

    Returns hazard ∈ [0, ∞) (typically [0, 2])
    """
    brittleness_factor = max(0, 1 - zeta / zeta_star)
    hazard = kappa * epsilon * g_phi * brittleness_factor * u * p

    return max(0, hazard)  # Ensure non-negative


# ============================================================================
# VBC Trade Decoder
# ============================================================================

class VBCTradeDecoder:
    """
    Variable Barrier Controller for trade selection.

    Decoder analogy:
    - Input: List of trade candidates (tokens)
    - Output: Ranked trades by hazard (logits)
    - Selection: Top-K or threshold-based (sampling)

    Enforces:
    - Risk limits (via ε)
    - Portfolio constraints (via ζ)
    - Signal alignment (via u)
    - Evidence requirements (via p)
    """

    def __init__(
        self,
        max_position_fraction: float = 0.1,
        zeta_star: float = 1.0,
        hazard_threshold: float = 0.1
    ):
        """
        Initialize VBC decoder.

        Args:
            max_position_fraction: Max position size as fraction of portfolio
            zeta_star: Brittleness threshold
            hazard_threshold: Minimum hazard to consider trade
        """
        self.max_position_fraction = max_position_fraction
        self.zeta_star = zeta_star
        self.hazard_threshold = hazard_threshold

    def create_trade_candidate(
        self,
        state: DeltaState,
        lock: LockState,
        symbol: str,
        action: str,
        quantity: float,
        entry_price: float
    ) -> TradeCandidate:
        """
        Create a trade candidate with computed hazard.

        Args:
            state: DeltaState
            lock: LockState generating this trade
            symbol: Trading symbol
            action: "BUY", "SELL", "HOLD"
            quantity: Position size
            entry_price: Entry price

        Returns:
            TradeCandidate with computed hazard
        """
        # Compute hazard components
        kappa = compute_kappa(
            expected_return=lock.K * 0.01,  # Simplified
            volatility=0.01  # Placeholder
        )

        epsilon = compute_epsilon(
            portfolio=state.portfolio,
            regime=state.regime,
            position_size=quantity * entry_price,
            max_position_fraction=self.max_position_fraction
        )

        g_phi = compute_g_phi(
            lock=lock,
            current_phase_diff=0.0  # Placeholder - would compute from prices
        )

        zeta = compute_zeta_trade(
            portfolio=state.portfolio,
            symbol=symbol,
            position_size=quantity * entry_price
        )

        # Signal alignment (would get from consensus detector)
        signal_strengths = {"lock": lock.K}  # Simplified
        u = compute_u_alignment(action, signal_strengths)

        p = compute_p_prior(lock)

        # Compute hazard
        hazard = compute_hazard_canonical(
            kappa, epsilon, g_phi, zeta, u, p, self.zeta_star
        )

        # Create candidate
        candidate = TradeCandidate(
            trade_id=f"{symbol}_{action}_{datetime.utcnow().timestamp()}",
            lock_id=lock.id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=entry_price,
            kappa=kappa,
            epsilon=epsilon,
            g_phi=g_phi,
            zeta=zeta,
            u=u,
            p=p,
            hazard=hazard,
            expected_pnl=kappa * abs(quantity * entry_price) * 0.01,  # Simplified
            expected_delta_h=kappa * u * p  # Simplified
        )

        return candidate

    def generate_candidates_from_locks(
        self,
        state: DeltaState
    ) -> List[TradeCandidate]:
        """
        Generate trade candidates from all deployable locks.

        Args:
            state: DeltaState

        Returns:
            List of TradeCandidate objects
        """
        candidates = []

        # Get deployable locks
        deployable_locks = state.get_deployable_locks()

        for lock in deployable_locks:
            # Generate BUY candidate for asset A
            if lock.K > 0:  # Positive coupling
                candidate_buy = self.create_trade_candidate(
                    state=state,
                    lock=lock,
                    symbol=lock.a,
                    action="BUY",
                    quantity=100,  # Simplified - would compute optimal size
                    entry_price=100.0  # Placeholder - would get from market data
                )
                candidates.append(candidate_buy)

            # Generate SELL candidate for asset A
            elif lock.K < 0:  # Negative coupling
                candidate_sell = self.create_trade_candidate(
                    state=state,
                    lock=lock,
                    symbol=lock.a,
                    action="SELL",
                    quantity=100,
                    entry_price=100.0
                )
                candidates.append(candidate_sell)

        return candidates

    def rank_candidates(
        self,
        candidates: List[TradeCandidate]
    ) -> List[TradeCandidate]:
        """
        Rank candidates by hazard (descending).

        Args:
            candidates: List of TradeCandidate

        Returns:
            Ranked list
        """
        return sorted(candidates, key=lambda c: c.hazard, reverse=True)

    def select_top_k(
        self,
        candidates: List[TradeCandidate],
        k: int = 5
    ) -> List[TradeCandidate]:
        """
        Select top K candidates by hazard.

        Args:
            candidates: Ranked candidates
            k: Number to select

        Returns:
            Top K candidates
        """
        # Filter by hazard threshold
        eligible = [c for c in candidates if c.hazard >= self.hazard_threshold]

        # Take top K
        return eligible[:k]

    def decode(
        self,
        state: DeltaState,
        max_trades: int = 5
    ) -> List[TradeCandidate]:
        """
        Main decoder method: Generate and select trades.

        Args:
            state: DeltaState
            max_trades: Max number of trades to return

        Returns:
            Selected TradeCandidate objects
        """
        # Generate candidates
        candidates = self.generate_candidates_from_locks(state)

        if not candidates:
            state.add_log("VBC: No trade candidates generated")
            return []

        # Rank by hazard
        ranked = self.rank_candidates(candidates)

        # Select top K
        selected = self.select_top_k(ranked, k=max_trades)

        state.add_log(f"VBC: Generated {len(candidates)} candidates, selected {len(selected)}")

        return selected

    def candidates_to_hazard_items(
        self,
        candidates: List[TradeCandidate]
    ) -> List[HazardItem]:
        """
        Convert TradeCandidate objects to HazardItem for state tracking.

        Args:
            candidates: List of TradeCandidate

        Returns:
            List of HazardItem
        """
        hazard_items = []

        for candidate in candidates:
            item = HazardItem(
                id=candidate.trade_id,
                category="trade",
                epsilon=candidate.epsilon,
                g_phi=candidate.g_phi,
                zeta=candidate.zeta,
                u=candidate.u,
                p=candidate.p,
                kappa=candidate.kappa,
                hazard=candidate.hazard,
                meta={
                    'symbol': candidate.symbol,
                    'action': candidate.action,
                    'quantity': candidate.quantity,
                    'entry_price': candidate.entry_price,
                    'lock_id': candidate.lock_id
                }
            )
            hazard_items.append(item)

        return hazard_items


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from delta_state_v2 import (
        create_research_state, RegimeState, PortfolioState, Position,
        OperatingMode, EStatus
    )

    print("=" * 70)
    print("VBC TRADE DECODER: Hazard-Based Trade Selection")
    print("=" * 70)

    # Create state
    state = create_research_state()
    state.operating_mode = OperatingMode.MICRO_LIVE

    # Add portfolio
    state.portfolio = PortfolioState(
        timestamp=datetime.utcnow(),
        cash=100000.0  # $100K for testing
    )

    # Add regime
    state.regime = RegimeState(
        timestamp=datetime.utcnow(),
        global_regime=RegimeLabel.NORMAL,
        chi_global=0.4,
        chi_trend=0.38
    )

    # Add deployable lock
    lock = LockState(
        id="AAPL-MSFT-2:3",
        a="AAPL", b="MSFT",
        p=2, q=3,
        K=0.75,
        Gamma_a=0.05, Gamma_b=0.05,
        Q_a=15, Q_b=15,
        eps_cap=0.8, eps_stab=0.7, zeta=0.3,
        evidence_score=0.25
    )
    lock.e0_status = EStatus.PASS
    lock.e1_status = EStatus.PASS
    lock.e2_status = EStatus.PASS
    lock.e3_status = EStatus.PASS
    lock.e_level_passed = 3

    state.locks[lock.id] = lock

    # Verify lock is deployable
    print(f"\n[LOCK STATUS]")
    print(f"  Is potential: {lock.is_potential()}")
    print(f"  Is actualized: {lock.is_actualized()}")
    print(f"  Is deployable: {lock.is_deployable()}")
    print(f"  Deployable locks in state: {len(state.get_deployable_locks())}")

    # Create decoder
    decoder = VBCTradeDecoder(
        max_position_fraction=0.10,
        zeta_star=1.0,
        hazard_threshold=0.05
    )

    # Decode trades
    print(f"\n[DECODING TRADES]")

    # Debug: Generate candidates manually
    candidates = decoder.generate_candidates_from_locks(state)
    print(f"Generated {len(candidates)} candidates")
    for c in candidates:
        print(f"  {c.symbol} {c.action}: hazard={c.hazard:.3f}, epsilon={c.epsilon:.3f}")

    selected_trades = decoder.decode(state, max_trades=5)

    print(f"\nSelected {len(selected_trades)} trades:")
    for i, trade in enumerate(selected_trades, 1):
        print(f"\n{i}. {trade.symbol} {trade.action}")
        print(f"   Quantity: {trade.quantity}")
        print(f"   Hazard: {trade.hazard:.3f}")
        print(f"   Components: κ={trade.kappa:.3f}, ε={trade.epsilon:.3f}, "
              f"g={trade.g_phi:.3f}, ζ={trade.zeta:.3f}, u={trade.u:.3f}, p={trade.p:.3f}")
        print(f"   Expected P&L: ${trade.expected_pnl:.2f}")

    # Convert to HazardItems for state tracking
    hazard_items = decoder.candidates_to_hazard_items(selected_trades)
    state.hazards.extend(hazard_items)

    print(f"\n[STATE UPDATE]")
    print(f"Added {len(hazard_items)} hazard items to state.hazards")
    print(f"Total hazards tracked: {len(state.hazards)}")

    print("\n" + "=" * 70)
    print("VBC: Same decoder physics for LLM tokens and trading actions")
    print("=" * 70)
