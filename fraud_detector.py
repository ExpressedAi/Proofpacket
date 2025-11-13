"""
S* Fraud Detection via Cross-Structure Phase-Lock Analysis
Layer 3: Risk Filter for Δ-Trading System

This module detects potential fraud by measuring phase-lock coherence across
multiple data structures (price, volume, fundamentals, insider activity).

Key Insight:
    Healthy companies maintain consistent phase-locks across:
        - Price ↔ Volume
        - Price ↔ Executive Compensation
        - Volume ↔ Audit Fees
        - Compensation ↔ Audit Fees

    When these decouple, it signals potential manipulation/fraud.

S* Score Formula:
    S* = w_K·K_avg - w_ζ·ζ_avg - w_χ·χ²_symmetry - w_KL·D_KL

    where:
        K_avg = average coupling strength across pairs
        ζ_avg = average brittleness across pairs
        χ²_symmetry = symmetry residual (should be ~0 for honest firms)
        D_KL = KL divergence from null model

Risk Signal:
    S* z-score < -2.5 for 5+ consecutive days → EXCLUDE from universe
    This indicates systematic decoupling (fraud indicator)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy.stats import zscore as compute_zscore
from scipy.signal import hilbert


@dataclass
class FraudSignal:
    """Fraud detection signal for a stock."""
    ticker: str
    S_star: float           # Unified fraud score
    S_star_zscore: float    # Z-score relative to universe
    is_suspicious: bool     # True if S* < threshold
    days_suspicious: int    # Consecutive days below threshold
    breakdown: Dict[str, float]  # Component scores

    def __str__(self):
        status = "⚠️ SUSPICIOUS" if self.is_suspicious else "✓ Clean"
        return f"{self.ticker}: S*={self.S_star:.2f} (z={self.S_star_zscore:.2f}) [{status}]"


class CrossStructureData:
    """
    Container for multi-structure data needed for S* calculation.

    In production, this would pull from multiple sources:
        - Price/Volume: Market data API
        - Executive Comp: SEC filings (DEF14A)
        - Audit Fees: 10-K filings
        - Insider Trading: SEC Form 4
    """

    def __init__(self, ticker: str):
        self.ticker = ticker

        # Market data (time series)
        self.price: np.ndarray = np.array([])
        self.volume: np.ndarray = np.array([])

        # Fundamental data (time series or point-in-time)
        self.exec_comp: np.ndarray = np.array([])  # Executive compensation
        self.audit_fees: np.ndarray = np.array([])  # Audit fees
        self.insider_trades: np.ndarray = np.array([])  # Insider trade volume

        # Derived metrics
        self.revenue: np.ndarray = np.array([])
        self.earnings: np.ndarray = np.array([])

    def add_market_data(self, price: np.ndarray, volume: np.ndarray):
        """Add price and volume time series."""
        self.price = price
        self.volume = volume

    def add_fundamental_data(
        self,
        exec_comp: np.ndarray,
        audit_fees: np.ndarray,
        revenue: np.ndarray,
        earnings: np.ndarray
    ):
        """Add fundamental data time series (quarterly or annual)."""
        self.exec_comp = exec_comp
        self.audit_fees = audit_fees
        self.revenue = revenue
        self.earnings = earnings

    def has_sufficient_data(self, min_length: int = 20) -> bool:
        """Check if we have enough data for analysis."""
        return (
            len(self.price) >= min_length and
            len(self.volume) >= min_length and
            len(self.exec_comp) >= 4 and  # At least 4 quarters
            len(self.audit_fees) >= 4
        )


class FraudDetector:
    """
    Detects fraud via cross-structure phase-lock analysis.

    Algorithm:
        1. Compute phase-locks between all pairs of metrics
        2. Calculate average coupling K and brittleness ζ
        3. Measure symmetry violations (χ² test)
        4. Compute KL divergence from null model
        5. Combine into S* unified score
        6. Flag stocks with S* z-score < -2.5 for 5+ days
    """

    def __init__(
        self,
        z_threshold: float = -2.5,    # Z-score threshold for flagging
        consecutive_days: int = 5,     # Days below threshold to flag
        w_K: float = 1.0,              # Weight for coupling strength
        w_zeta: float = 0.5,           # Weight for brittleness penalty
        w_chi: float = 0.3,            # Weight for symmetry violation
        w_KL: float = 0.2,             # Weight for KL divergence
    ):
        """
        Args:
            z_threshold: Z-score threshold for flagging (negative)
            consecutive_days: Days below threshold to exclude
            w_K: Weight for coupling strength (higher = better)
            w_zeta: Weight for brittleness penalty (higher = worse)
            w_chi: Weight for symmetry violation (higher = worse)
            w_KL: Weight for KL divergence (higher = worse)
        """
        self.z_threshold = z_threshold
        self.consecutive_days = consecutive_days
        self.w_K = w_K
        self.w_zeta = w_zeta
        self.w_chi = w_chi
        self.w_KL = w_KL

        # History tracking
        self.S_star_history: Dict[str, deque] = {}
        self.zscore_history: Dict[str, deque] = {}
        self.suspicious_days: Dict[str, int] = {}

    def compute_phase_lock_strength(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        window: int = 20
    ) -> Tuple[float, float]:
        """
        Compute phase-lock strength K and brittleness ζ for a pair.

        Returns:
            (K, ζ) where:
                K ∈ [-1, 1]: Coupling strength (1 = perfect lock)
                ζ ∈ [0, ∞): Brittleness (0 = robust, high = brittle)
        """
        if len(series_a) < window or len(series_b) < window:
            return 0.0, 1.0  # Default to weak, brittle

        # Normalize
        a = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-10)
        b = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-10)

        # Hilbert transform for phase extraction
        analytic_a = hilbert(a)
        analytic_b = hilbert(b)

        phase_a = np.angle(analytic_a)
        phase_b = np.angle(analytic_b)

        # Phase difference
        delta_phi = phase_a - phase_b

        # Coupling strength K = <cos(Δφ)>
        K = np.mean(np.cos(delta_phi[-window:]))

        # Brittleness ζ = std(Δφ)
        zeta = np.std(delta_phi[-window:])

        return K, zeta

    def compute_symmetry_residual(
        self,
        price: np.ndarray,
        volume: np.ndarray,
        revenue: np.ndarray
    ) -> float:
        """
        Compute χ² symmetry residual.

        Healthy firms: Price ∝ Revenue ∝ Volume (power-law relationships)
        Fraudulent firms: Relationships break down

        Returns:
            χ² ∈ [0, ∞) where 0 = perfect symmetry
        """
        if len(price) < 10 or len(volume) < 10 or len(revenue) < 4:
            return 0.0

        # Align to quarterly data (subsample price/volume)
        n_quarters = len(revenue)
        quarter_length = len(price) // n_quarters

        price_quarterly = []
        volume_quarterly = []

        for i in range(n_quarters):
            start = i * quarter_length
            end = (i + 1) * quarter_length
            if end <= len(price):
                price_quarterly.append(np.mean(price[start:end]))
                volume_quarterly.append(np.mean(volume[start:end]))

        if len(price_quarterly) < 4:
            return 0.0

        price_q = np.array(price_quarterly)
        volume_q = np.array(volume_quarterly)

        # Expected relationships (log-space)
        # Use absolute values to handle any negative values
        log_price = np.log(np.abs(price_q) + 1)
        log_volume = np.log(np.abs(volume_q) + 1)
        log_revenue = np.log(np.abs(revenue) + 1)

        # Correlation matrix
        data = np.column_stack([log_price, log_volume, log_revenue])
        corr_matrix = np.corrcoef(data.T)

        # For symmetric system, all correlations should be similar
        # χ² = variance of correlation coefficients
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        chi_squared = np.var(upper_triangle)

        return chi_squared

    def compute_kl_divergence(
        self,
        observed_K: np.ndarray,
        null_mean: float = 0.3,
        null_std: float = 0.2
    ) -> float:
        """
        Compute KL divergence of observed K distribution vs null model.

        Null model: K ~ N(0.3, 0.2) (typical cross-asset coupling)
        Fraudulent: K distribution diverges from null

        Returns:
            D_KL ∈ [0, ∞) where 0 = matches null perfectly
        """
        if len(observed_K) < 3:
            return 0.0

        # Observed distribution
        obs_mean = np.mean(observed_K)
        obs_std = np.std(observed_K) + 1e-10

        # KL divergence for Gaussians:
        # D_KL(P||Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²)/(2σ_Q²) - 1/2
        D_KL = (
            np.log(null_std / obs_std) +
            (obs_std**2 + (obs_mean - null_mean)**2) / (2 * null_std**2) -
            0.5
        )

        return max(0.0, D_KL)

    def compute_S_star(self, data: CrossStructureData) -> Tuple[float, Dict[str, float]]:
        """
        Compute unified fraud score S* for a stock.

        Returns:
            (S*, breakdown) where breakdown contains component scores
        """
        if not data.has_sufficient_data():
            return 0.0, {}

        # Define pairs to analyze
        pairs = []
        K_values = []
        zeta_values = []

        # Pair 1: Price ↔ Volume
        K_pv, zeta_pv = self.compute_phase_lock_strength(data.price, data.volume)
        pairs.append(("price", "volume", K_pv, zeta_pv))
        K_values.append(K_pv)
        zeta_values.append(zeta_pv)

        # Pair 2: Price ↔ Earnings (quarterly)
        if len(data.earnings) >= 4:
            # Subsample price to quarterly
            n_quarters = len(data.earnings)
            quarter_length = len(data.price) // n_quarters
            price_q = []
            for i in range(n_quarters):
                start = i * quarter_length
                end = (i + 1) * quarter_length
                if end <= len(data.price):
                    price_q.append(np.mean(data.price[start:end]))

            if len(price_q) == len(data.earnings):
                K_pe, zeta_pe = self.compute_phase_lock_strength(
                    np.array(price_q), data.earnings
                )
                pairs.append(("price", "earnings", K_pe, zeta_pe))
                K_values.append(K_pe)
                zeta_values.append(zeta_pe)

        # Pair 3: Executive Comp ↔ Audit Fees
        if len(data.exec_comp) >= 4 and len(data.audit_fees) >= 4:
            min_len = min(len(data.exec_comp), len(data.audit_fees))
            K_ca, zeta_ca = self.compute_phase_lock_strength(
                data.exec_comp[:min_len], data.audit_fees[:min_len]
            )
            pairs.append(("comp", "audit", K_ca, zeta_ca))
            K_values.append(K_ca)
            zeta_values.append(zeta_ca)

        # Pair 4: Revenue ↔ Audit Fees (should scale together)
        if len(data.revenue) >= 4 and len(data.audit_fees) >= 4:
            min_len = min(len(data.revenue), len(data.audit_fees))
            K_ra, zeta_ra = self.compute_phase_lock_strength(
                data.revenue[:min_len], data.audit_fees[:min_len]
            )
            pairs.append(("revenue", "audit", K_ra, zeta_ra))
            K_values.append(K_ra)
            zeta_values.append(zeta_ra)

        # Average coupling and brittleness
        K_avg = np.mean(K_values) if K_values else 0.0
        zeta_avg = np.mean(zeta_values) if zeta_values else 1.0

        # Symmetry residual
        chi_squared = self.compute_symmetry_residual(
            data.price, data.volume, data.revenue
        )

        # KL divergence
        D_KL = self.compute_kl_divergence(np.array(K_values))

        # S* = weighted combination
        S_star = (
            self.w_K * K_avg -
            self.w_zeta * zeta_avg -
            self.w_chi * chi_squared -
            self.w_KL * D_KL
        )

        breakdown = {
            'K_avg': K_avg,
            'zeta_avg': zeta_avg,
            'chi_squared': chi_squared,
            'D_KL': D_KL,
            'S_star': S_star
        }

        return S_star, breakdown

    def update(
        self,
        ticker: str,
        data: CrossStructureData,
        universe_S_stars: Optional[List[float]] = None
    ) -> FraudSignal:
        """
        Update fraud signal for a stock.

        Args:
            ticker: Stock ticker
            data: Cross-structure data
            universe_S_stars: S* values for entire universe (for z-score)

        Returns:
            FraudSignal with current status
        """
        # Compute S*
        S_star, breakdown = self.compute_S_star(data)

        # Initialize history if needed
        if ticker not in self.S_star_history:
            self.S_star_history[ticker] = deque(maxlen=30)
            self.zscore_history[ticker] = deque(maxlen=30)
            self.suspicious_days[ticker] = 0

        # Add to history
        self.S_star_history[ticker].append(S_star)

        # Compute z-score relative to universe
        if universe_S_stars is not None and len(universe_S_stars) > 10:
            all_scores = np.array(universe_S_stars + [S_star])
            zscores = compute_zscore(all_scores)
            S_star_zscore = zscores[-1]
        else:
            # Fallback: z-score relative to own history
            if len(self.S_star_history[ticker]) > 5:
                history = np.array(list(self.S_star_history[ticker]))
                S_star_zscore = (S_star - np.mean(history)) / (np.std(history) + 1e-10)
            else:
                S_star_zscore = 0.0

        self.zscore_history[ticker].append(S_star_zscore)

        # Check if suspicious
        is_suspicious = S_star_zscore < self.z_threshold

        # Update consecutive days counter
        if is_suspicious:
            self.suspicious_days[ticker] += 1
        else:
            self.suspicious_days[ticker] = 0

        days_suspicious = self.suspicious_days[ticker]

        # Flag if suspicious for consecutive_days threshold
        should_exclude = days_suspicious >= self.consecutive_days

        return FraudSignal(
            ticker=ticker,
            S_star=S_star,
            S_star_zscore=S_star_zscore,
            is_suspicious=should_exclude,
            days_suspicious=days_suspicious,
            breakdown=breakdown
        )

    def filter_universe(
        self,
        universe: List[str],
        data_map: Dict[str, CrossStructureData]
    ) -> List[str]:
        """
        Filter universe to exclude suspicious stocks.

        Args:
            universe: List of tickers
            data_map: Map of ticker -> CrossStructureData

        Returns:
            Filtered list of tickers (excluding suspicious ones)
        """
        # Compute S* for all stocks
        S_stars = []
        signals = []

        for ticker in universe:
            if ticker in data_map:
                data = data_map[ticker]
                S_star, _ = self.compute_S_star(data)
                S_stars.append(S_star)
            else:
                S_stars.append(0.0)

        # Update all signals with universe context
        for ticker in universe:
            if ticker in data_map:
                signal = self.update(ticker, data_map[ticker], universe_S_stars=S_stars)
                signals.append(signal)

        # Filter out suspicious stocks
        clean_universe = [
            ticker for ticker, signal in zip(universe, signals)
            if not signal.is_suspicious
        ]

        return clean_universe


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("S* Fraud Detector - Example Usage")
    print("=" * 70)

    # Create detector
    detector = FraudDetector(
        z_threshold=-2.5,
        consecutive_days=5
    )

    print(f"\nConfiguration:")
    print(f"  Z-score threshold: {detector.z_threshold}")
    print(f"  Consecutive days: {detector.consecutive_days}")
    print(f"  Weights: K={detector.w_K}, ζ={detector.w_zeta}, "
          f"χ²={detector.w_chi}, D_KL={detector.w_KL}")

    # Generate synthetic data for healthy and fraudulent companies
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("Synthetic Company Analysis")
    print("=" * 70)

    # Healthy Company
    print("\n1. HEALTHY CORP (HLTHY)")
    print("-" * 40)

    healthy = CrossStructureData("HLTHY")

    # Generate correlated price/volume (phase-locked)
    t = np.arange(100)
    base_signal = np.sin(2 * np.pi * t / 20)
    healthy.price = 100 + 10 * base_signal + np.random.normal(0, 1, 100)
    healthy.volume = 1e6 + 1e5 * base_signal + np.random.normal(0, 1e4, 100)

    # Quarterly data (correlated)
    quarters = 8
    base_q = np.linspace(1, 2, quarters)
    healthy.revenue = 1e9 * base_q + np.random.normal(0, 1e7, quarters)
    healthy.earnings = 1e8 * base_q + np.random.normal(0, 1e6, quarters)
    healthy.exec_comp = 1e6 * base_q + np.random.normal(0, 1e4, quarters)
    healthy.audit_fees = 1e5 * base_q + np.random.normal(0, 1e3, quarters)

    S_star_healthy, breakdown_healthy = detector.compute_S_star(healthy)

    print(f"  S* Score: {S_star_healthy:.3f}")
    print(f"  Components:")
    for key, value in breakdown_healthy.items():
        print(f"    {key:12s}: {value:7.3f}")

    # Fraudulent Company
    print("\n2. FRAUD INC (FRОД)")
    print("-" * 40)

    fraud = CrossStructureData("FRОД")

    # Price and volume DECOUPLED (red flag)
    fraud.price = 100 + 20 * np.sin(2 * np.pi * t / 15) + np.random.normal(0, 5, 100)
    fraud.volume = 1e6 + 1e5 * np.cos(2 * np.pi * t / 30) + np.random.normal(0, 5e4, 100)

    # Quarterly: Revenue declining but exec comp INCREASING (red flag)
    fraud.revenue = 1e9 * (1.5 - 0.3 * base_q) + np.random.normal(0, 1e7, quarters)  # Declining
    fraud.earnings = 1e8 * (1.3 - 0.2 * base_q) + np.random.normal(0, 5e6, quarters)  # Declining
    fraud.exec_comp = 2e6 * base_q + np.random.normal(0, 1e5, quarters)  # INCREASING (red flag!)
    fraud.audit_fees = 5e4 * np.ones(quarters) + np.random.normal(0, 1e3, quarters)  # Flat (suspicious)

    S_star_fraud, breakdown_fraud = detector.compute_S_star(fraud)

    print(f"  S* Score: {S_star_fraud:.3f}")
    print(f"  Components:")
    for key, value in breakdown_fraud.items():
        print(f"    {key:12s}: {value:7.3f}")

    # Compare
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"\nS* Scores:")
    print(f"  HEALTHY: {S_star_healthy:7.3f} (higher = better)")
    print(f"  FRAUD:   {S_star_fraud:7.3f}")
    print(f"  Δ:       {S_star_healthy - S_star_fraud:7.3f}")

    if S_star_healthy > S_star_fraud:
        print(f"\n✓ Detector correctly identifies FRAUD as more suspicious")
    else:
        print(f"\n✗ Detector failed (healthy scored lower)")

    # Universe filtering example
    print("\n" + "=" * 70)
    print("Universe Filtering Example")
    print("=" * 70)

    universe = ["HLTHY", "FRОД", "AAPL", "MSFT", "TSLA"]
    data_map = {
        "HLTHY": healthy,
        "FRОД": fraud,
    }

    # Add more synthetic companies
    for ticker in ["AAPL", "MSFT", "TSLA"]:
        company = CrossStructureData(ticker)
        company.price = 150 + 15 * np.sin(2 * np.pi * t / 22) + np.random.normal(0, 2, 100)
        company.volume = 2e6 + 2e5 * np.sin(2 * np.pi * t / 22) + np.random.normal(0, 2e4, 100)
        company.revenue = 2e9 * base_q + np.random.normal(0, 2e7, quarters)
        company.earnings = 2e8 * base_q + np.random.normal(0, 2e6, quarters)
        company.exec_comp = 1.5e6 * base_q + np.random.normal(0, 1e4, quarters)
        company.audit_fees = 1.2e5 * base_q + np.random.normal(0, 1e3, quarters)
        data_map[ticker] = company

    print(f"\nOriginal Universe: {universe}")

    # Filter
    clean_universe = detector.filter_universe(universe, data_map)

    print(f"Clean Universe:    {clean_universe}")
    print(f"Excluded:          {set(universe) - set(clean_universe)}")

    print("\n" + "=" * 70)
    print("✓ Day 3 implementation complete")
    print("  S* fraud detector ready for integration")
    print("  Next: Day 4 - TUR execution optimizer")
    print("=" * 70)
