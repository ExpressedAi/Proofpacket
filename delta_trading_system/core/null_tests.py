"""
Null Tests: Domain-Specific Null Hypotheses for E1 Gates

Goal: Each layer beats its own nulls before we trust it.

Implements null families for:
- Layer 1 (Consensus): Label shuffle, block bootstrap, simple benchmark
- Layer 2 (χ-crash): Vol-only, randomized regime, phase-shifted
- Layer 3 (S* Fraud): Structure-randomized, Gaussian K-null, healthy-only
- Layer 4 (TUR): Random rebalancing, equal-weight, Sharpe-only

All nulls return (null_distribution, p_value, fdr_corrected_p).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Any, Optional
import numpy as np
from scipy import stats
from datetime import datetime


# ============================================================================
# Null Test Result Structure
# ============================================================================

@dataclass
class NullTestResult:
    """Result of a null hypothesis test."""
    layer: str  # "consensus", "chi_crash", "fraud", "tur"
    null_type: str  # "label_shuffle", "vol_only", etc.

    # Observed vs null distribution
    observed_stat: float
    null_distribution: np.ndarray

    # Statistical results
    p_value: float
    fdr_corrected_p: float  # Benjamini-Hochberg

    # Metadata
    n_surrogates: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    passed: bool = False  # p < 0.05 after FDR
    notes: str = ""


# ============================================================================
# Generic Null Generators
# ============================================================================

def generate_phase_shuffle_null(data: np.ndarray, n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Phase shuffle null: Preserve amplitude spectrum, randomize phases.

    Destroys temporal correlations but keeps power spectrum.
    Good for testing if structure is just noise.
    """
    surrogates = []

    for _ in range(n_surrogates):
        fft = np.fft.fft(data)
        amplitudes = np.abs(fft)

        # Randomize phases
        random_phases = np.random.uniform(0, 2 * np.pi, len(fft))
        surrogate_fft = amplitudes * np.exp(1j * random_phases)

        # Inverse FFT
        surrogate = np.real(np.fft.ifft(surrogate_fft))
        surrogates.append(surrogate)

    return surrogates


def generate_block_surrogate(data: np.ndarray, block_size: int = 20,
                             n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Block surrogate: Resample in blocks to preserve local dynamics.

    Destroys long-range correlations but keeps short-term structure.
    """
    surrogates = []
    n = len(data)
    n_blocks = n // block_size

    for _ in range(n_surrogates):
        # Create blocks
        blocks = [data[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

        # Shuffle blocks
        np.random.shuffle(blocks)

        # Concatenate
        surrogate = np.concatenate(blocks)

        # Pad if necessary
        if len(surrogate) < n:
            surrogate = np.concatenate([surrogate, data[len(surrogate):]])

        surrogates.append(surrogate)

    return surrogates


def generate_label_shuffle(labels: np.ndarray, n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Label shuffle: Randomly permute labels.

    Tests if signal-label association is real or random.
    """
    surrogates = []

    for _ in range(n_surrogates):
        shuffled = labels.copy()
        np.random.shuffle(shuffled)
        surrogates.append(shuffled)

    return surrogates


def generate_vol_matched_null(data: np.ndarray, n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Vol-matched null: Random walk with same volatility.

    Tests if structure is just scaled noise.
    """
    surrogates = []
    vol = np.std(np.diff(data))

    for _ in range(n_surrogates):
        returns = np.random.normal(0, vol, len(data) - 1)
        surrogate = np.concatenate([[data[0]], data[0] + np.cumsum(returns)])
        surrogates.append(surrogate)

    return surrogates


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Returns list of booleans: True if hypothesis is rejected (significant).
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute critical values
    critical_values = (np.arange(1, n + 1) / n) * alpha

    # Find largest i where p[i] <= (i/n)*alpha
    rejections = sorted_p <= critical_values

    if not np.any(rejections):
        return [False] * n

    # Find threshold
    threshold_idx = np.where(rejections)[0][-1]
    threshold_p = sorted_p[threshold_idx]

    # Reject all p-values <= threshold
    results = [False] * n
    for i in range(n):
        if p_values[i] <= threshold_p:
            results[i] = True

    return results


# ============================================================================
# Layer 1: Consensus / Signal Layer Nulls
# ============================================================================

def consensus_label_shuffle_null(
    signals: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Label shuffle null for consensus detector.

    Tests if signal->label mapping is better than random.

    Args:
        signals: (N, M) array of M signals over N timesteps
        labels: (N,) array of true labels (e.g., BUY/SELL)
        metric_fn: Function computing metric (e.g., accuracy, F1)
    """
    observed_stat = metric_fn(signals, labels)

    # Generate null distribution
    null_stats = []
    for _ in range(n_surrogates):
        shuffled_labels = labels.copy()
        np.random.shuffle(shuffled_labels)
        null_stat = metric_fn(signals, shuffled_labels)
        null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # Compute p-value (one-sided: observed > null)
    p_value = np.mean(null_distribution >= observed_stat)

    # FDR correction (single test, so just pass through)
    fdr_corrected_p = p_value

    return NullTestResult(
        layer="consensus",
        null_type="label_shuffle",
        observed_stat=observed_stat,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=fdr_corrected_p,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"Observed={observed_stat:.3f}, p={p_value:.3f}"
    )


def consensus_block_bootstrap_null(
    signals: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    block_size: int = 20,
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Block bootstrap null for consensus detector.

    Tests if performance is robust to temporal resampling.
    """
    observed_stat = metric_fn(signals, labels)

    n = len(signals)
    n_blocks = n // block_size

    # Generate null distribution
    null_stats = []
    for _ in range(n_surrogates):
        # Resample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        bootstrap_signals = []
        bootstrap_labels = []

        for idx in block_indices:
            start = idx * block_size
            end = start + block_size
            bootstrap_signals.append(signals[start:end])
            bootstrap_labels.append(labels[start:end])

        bootstrap_signals = np.concatenate(bootstrap_signals)
        bootstrap_labels = np.concatenate(bootstrap_labels)

        null_stat = metric_fn(bootstrap_signals, bootstrap_labels)
        null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # Compute p-value (two-sided: test if observed is unusual)
    p_value_lower = np.mean(null_distribution <= observed_stat)
    p_value_upper = np.mean(null_distribution >= observed_stat)
    p_value = 2 * min(p_value_lower, p_value_upper)

    return NullTestResult(
        layer="consensus",
        null_type="block_bootstrap",
        observed_stat=observed_stat,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"Observed={observed_stat:.3f}, p={p_value:.3f}"
    )


def consensus_simple_benchmark_null(
    signals: np.ndarray,
    labels: np.ndarray,
    observed_accuracy: float
) -> NullTestResult:
    """
    Simple benchmark null: Compare to naive strategies.

    Benchmarks:
    - Random: 50% accuracy
    - Always buy: Accuracy = fraction of buy labels
    - Always sell: Accuracy = fraction of sell labels
    """
    n = len(labels)

    # Random baseline
    random_accuracy = 0.5

    # Always-buy baseline
    buy_fraction = np.mean(labels > 0)
    always_buy_accuracy = buy_fraction

    # Always-sell baseline
    sell_fraction = np.mean(labels <= 0)
    always_sell_accuracy = sell_fraction

    # Best baseline
    best_baseline = max(random_accuracy, always_buy_accuracy, always_sell_accuracy)

    # Observed beats baseline?
    beats_baseline = observed_accuracy > best_baseline

    # Pseudo p-value (not statistical, just threshold-based)
    p_value = 0.01 if beats_baseline else 0.99

    return NullTestResult(
        layer="consensus",
        null_type="simple_benchmark",
        observed_stat=observed_accuracy,
        null_distribution=np.array([random_accuracy, always_buy_accuracy, always_sell_accuracy]),
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=3,
        passed=beats_baseline,
        notes=f"Observed={observed_accuracy:.3f}, Best baseline={best_baseline:.3f}"
    )


# ============================================================================
# Layer 2: χ-Crash Detector Nulls
# ============================================================================

def chi_vol_only_null(
    prices: np.ndarray,
    chi_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Vol-only null for χ-crash detector.

    Tests if χ signal is just volatility in disguise.
    Generates random walks with same volatility.
    """
    observed_chi = chi_fn(prices)

    # Generate vol-matched nulls
    vol = np.std(np.diff(np.log(prices)))

    null_chis = []
    for _ in range(n_surrogates):
        log_returns = np.random.normal(0, vol, len(prices) - 1)
        null_prices = prices[0] * np.exp(np.cumsum(log_returns))
        null_chi = chi_fn(null_prices)
        null_chis.append(null_chi)

    null_distribution = np.array(null_chis)

    # One-sided: observed χ should be HIGHER than vol-only
    p_value = np.mean(null_distribution >= observed_chi)

    return NullTestResult(
        layer="chi_crash",
        null_type="vol_only",
        observed_stat=observed_chi,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"χ_obs={observed_chi:.3f}, χ_null_mean={np.mean(null_distribution):.3f}"
    )


def chi_randomized_regime_null(
    prices: np.ndarray,
    regime_labels: np.ndarray,
    chi_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Randomized regime null for χ-crash detector.

    Tests if regime classification is better than random.
    """
    # Compute observed χ in each regime
    regimes = np.unique(regime_labels)
    observed_chi_by_regime = {}

    for regime in regimes:
        mask = (regime_labels == regime)
        if np.sum(mask) > 10:  # Need enough data
            observed_chi_by_regime[regime] = chi_fn(prices[mask])

    # Metric: variance of χ across regimes (should be high if regimes distinct)
    observed_stat = np.var(list(observed_chi_by_regime.values()))

    # Generate null by shuffling regime labels
    null_stats = []
    for _ in range(n_surrogates):
        shuffled_labels = regime_labels.copy()
        np.random.shuffle(shuffled_labels)

        null_chi_by_regime = {}
        for regime in regimes:
            mask = (shuffled_labels == regime)
            if np.sum(mask) > 10:
                null_chi_by_regime[regime] = chi_fn(prices[mask])

        null_stat = np.var(list(null_chi_by_regime.values()))
        null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # One-sided: observed variance should be HIGHER
    p_value = np.mean(null_distribution >= observed_stat)

    return NullTestResult(
        layer="chi_crash",
        null_type="randomized_regime",
        observed_stat=observed_stat,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"Regime χ variance: obs={observed_stat:.3f}, null_mean={np.mean(null_distribution):.3f}"
    )


def chi_phase_shifted_null(
    prices: np.ndarray,
    chi_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Phase-shifted null for χ-crash detector.

    Tests if χ signal has meaningful phase structure.
    """
    observed_chi = chi_fn(prices)

    # Generate phase-shuffled nulls
    surrogates = generate_phase_shuffle_null(prices, n_surrogates)

    null_chis = []
    for surrogate in surrogates:
        null_chi = chi_fn(surrogate)
        null_chis.append(null_chi)

    null_distribution = np.array(null_chis)

    # One-sided: observed χ should be different (typically higher)
    p_value = np.mean(null_distribution >= observed_chi)

    return NullTestResult(
        layer="chi_crash",
        null_type="phase_shifted",
        observed_stat=observed_chi,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"χ_obs={observed_chi:.3f}, Phase-null mean={np.mean(null_distribution):.3f}"
    )


# ============================================================================
# Layer 3: S* Fraud Detector Nulls
# ============================================================================

def fraud_structure_randomized_null(
    coupling_matrix: np.ndarray,
    s_star_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Structure-randomized null for fraud detector.

    Tests if cross-structure phase-lock is real or random.
    Randomizes coupling matrix while preserving marginals.
    """
    observed_s_star = s_star_fn(coupling_matrix)

    # Generate null by randomizing off-diagonal elements
    null_stats = []
    n = coupling_matrix.shape[0]

    for _ in range(n_surrogates):
        # Preserve diagonal, shuffle off-diagonal
        null_matrix = coupling_matrix.copy()
        off_diag_indices = np.triu_indices(n, k=1)
        off_diag_values = null_matrix[off_diag_indices]
        np.random.shuffle(off_diag_values)

        null_matrix[off_diag_indices] = off_diag_values
        null_matrix = (null_matrix + null_matrix.T) / 2  # Symmetrize

        null_stat = s_star_fn(null_matrix)
        null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # One-sided: observed S* should be HIGHER
    p_value = np.mean(null_distribution >= observed_s_star)

    return NullTestResult(
        layer="fraud",
        null_type="structure_randomized",
        observed_stat=observed_s_star,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"S*_obs={observed_s_star:.3f}, Null mean={np.mean(null_distribution):.3f}"
    )


def fraud_gaussian_k_null(
    coupling_strengths: np.ndarray,
    threshold: float,
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Gaussian K-null for fraud detector.

    Tests if coupling strengths are stronger than Gaussian noise would produce.
    """
    observed_stat = np.mean(coupling_strengths > threshold)

    # Generate Gaussian nulls
    null_stats = []
    for _ in range(n_surrogates):
        null_K = np.random.normal(0, np.std(coupling_strengths), len(coupling_strengths))
        null_stat = np.mean(null_K > threshold)
        null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # One-sided: observed fraction should be HIGHER
    p_value = np.mean(null_distribution >= observed_stat)

    return NullTestResult(
        layer="fraud",
        null_type="gaussian_k_null",
        observed_stat=observed_stat,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"Fraction K>{threshold}: obs={observed_stat:.3f}, null_mean={np.mean(null_distribution):.3f}"
    )


def fraud_healthy_only_null(
    crisis_s_star: float,
    healthy_prices: np.ndarray,
    s_star_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Healthy-only null for fraud detector.

    Tests if crisis S* is significantly different from healthy periods.
    """
    # Generate null distribution from healthy periods
    null_stats = []
    window_size = len(healthy_prices) // n_surrogates

    for i in range(n_surrogates):
        start = i * window_size
        end = start + window_size
        if end <= len(healthy_prices):
            window = healthy_prices[start:end]
            null_stat = s_star_fn(window)
            null_stats.append(null_stat)

    null_distribution = np.array(null_stats)

    # One-sided: crisis S* should be HIGHER than healthy
    p_value = np.mean(null_distribution >= crisis_s_star)

    return NullTestResult(
        layer="fraud",
        null_type="healthy_only",
        observed_stat=crisis_s_star,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=len(null_stats),
        passed=(p_value < 0.05),
        notes=f"Crisis S*={crisis_s_star:.3f}, Healthy mean={np.mean(null_distribution):.3f}"
    )


# ============================================================================
# Layer 4: TUR / Execution Nulls
# ============================================================================

def tur_random_rebalancing_null(
    prices: np.ndarray,
    observed_sharpe: float,
    n_rebalances: int = 52,
    n_surrogates: int = 100
) -> NullTestResult:
    """
    Random rebalancing null for TUR optimizer.

    Tests if rebalancing schedule is better than random.
    """
    n = len(prices)

    # Generate null by random rebalancing
    null_sharpes = []
    for _ in range(n_surrogates):
        # Random rebalance times
        rebalance_times = np.sort(np.random.choice(n, size=n_rebalances, replace=False))

        # Compute returns with these rebalances
        returns = np.diff(np.log(prices))

        # Simple strategy: hold between rebalances
        null_returns = []
        for i in range(len(rebalance_times) - 1):
            start = rebalance_times[i]
            end = rebalance_times[i + 1]
            period_return = np.sum(returns[start:end])
            null_returns.append(period_return)

        null_returns = np.array(null_returns)
        null_sharpe = np.mean(null_returns) / (np.std(null_returns) + 1e-10) * np.sqrt(252)
        null_sharpes.append(null_sharpe)

    null_distribution = np.array(null_sharpes)

    # One-sided: observed Sharpe should be HIGHER
    p_value = np.mean(null_distribution >= observed_sharpe)

    return NullTestResult(
        layer="tur",
        null_type="random_rebalancing",
        observed_stat=observed_sharpe,
        null_distribution=null_distribution,
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=n_surrogates,
        passed=(p_value < 0.05),
        notes=f"Sharpe_obs={observed_sharpe:.3f}, Random_mean={np.mean(null_distribution):.3f}"
    )


def tur_equal_weight_null(
    portfolio_returns: np.ndarray,
    observed_sharpe: float
) -> NullTestResult:
    """
    Equal-weight null for TUR optimizer.

    Tests if dynamic optimization beats simple equal weighting.
    """
    # Equal-weight Sharpe
    equal_weight_sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(252)

    # Beats equal weight?
    beats_equal = observed_sharpe > equal_weight_sharpe

    # Pseudo p-value
    p_value = 0.01 if beats_equal else 0.99

    return NullTestResult(
        layer="tur",
        null_type="equal_weight",
        observed_stat=observed_sharpe,
        null_distribution=np.array([equal_weight_sharpe]),
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=1,
        passed=beats_equal,
        notes=f"Sharpe_obs={observed_sharpe:.3f}, Equal_weight={equal_weight_sharpe:.3f}"
    )


def tur_sharpe_only_null(
    observed_house_score: float,
    sharpe: float,
    max_dd: float
) -> NullTestResult:
    """
    Sharpe-only null for TUR optimizer.

    Tests if house score optimization beats naive Sharpe maximization.

    Computes what house score WOULD BE if we only cared about Sharpe
    (i.e., ignored crisis survival).
    """
    # Naive Sharpe-only would have higher returns but worse drawdown
    # Assume 2x Sharpe but 3x worse drawdown
    naive_sharpe = sharpe * 2
    naive_dd = max_dd * 3

    # Compute naive house score
    naive_crisis_survival = (1 - min(1.0, abs(naive_dd) / 0.55)) * 50
    naive_returns = min(naive_sharpe / 2.0, 1.0) * 10
    naive_house_score = naive_crisis_survival + naive_returns

    # Observed should beat naive
    beats_naive = observed_house_score > naive_house_score

    p_value = 0.01 if beats_naive else 0.99

    return NullTestResult(
        layer="tur",
        null_type="sharpe_only",
        observed_stat=observed_house_score,
        null_distribution=np.array([naive_house_score]),
        p_value=p_value,
        fdr_corrected_p=p_value,
        n_surrogates=1,
        passed=beats_naive,
        notes=f"House_obs={observed_house_score:.1f}, Naive={naive_house_score:.1f}"
    )


# ============================================================================
# Orchestrator: Run All Nulls for a Layer
# ============================================================================

def run_all_nulls_for_layer(
    layer: str,
    data: Dict[str, Any],
    alpha: float = 0.05
) -> Dict[str, NullTestResult]:
    """
    Run all null tests for a given layer.

    Returns dict of {null_type: NullTestResult}.
    """
    results = {}

    if layer == "consensus":
        # Requires: signals, labels, metric_fn
        if "signals" in data and "labels" in data and "metric_fn" in data:
            results["label_shuffle"] = consensus_label_shuffle_null(
                data["signals"], data["labels"], data["metric_fn"]
            )
            results["block_bootstrap"] = consensus_block_bootstrap_null(
                data["signals"], data["labels"], data["metric_fn"]
            )
            results["simple_benchmark"] = consensus_simple_benchmark_null(
                data["signals"], data["labels"], data["observed_accuracy"]
            )

    elif layer == "chi_crash":
        # Requires: prices, chi_fn
        if "prices" in data and "chi_fn" in data:
            results["vol_only"] = chi_vol_only_null(
                data["prices"], data["chi_fn"]
            )
            results["phase_shifted"] = chi_phase_shifted_null(
                data["prices"], data["chi_fn"]
            )
            if "regime_labels" in data:
                results["randomized_regime"] = chi_randomized_regime_null(
                    data["prices"], data["regime_labels"], data["chi_fn"]
                )

    elif layer == "fraud":
        # Requires: coupling_matrix, s_star_fn
        if "coupling_matrix" in data and "s_star_fn" in data:
            results["structure_randomized"] = fraud_structure_randomized_null(
                data["coupling_matrix"], data["s_star_fn"]
            )
        if "coupling_strengths" in data:
            results["gaussian_k_null"] = fraud_gaussian_k_null(
                data["coupling_strengths"], data.get("threshold", 0.5)
            )
        if "crisis_s_star" in data and "healthy_prices" in data and "s_star_fn" in data:
            results["healthy_only"] = fraud_healthy_only_null(
                data["crisis_s_star"], data["healthy_prices"], data["s_star_fn"]
            )

    elif layer == "tur":
        # Requires: prices, observed_sharpe
        if "prices" in data and "observed_sharpe" in data:
            results["random_rebalancing"] = tur_random_rebalancing_null(
                data["prices"], data["observed_sharpe"]
            )
        if "portfolio_returns" in data and "observed_sharpe" in data:
            results["equal_weight"] = tur_equal_weight_null(
                data["portfolio_returns"], data["observed_sharpe"]
            )
        if "observed_house_score" in data and "sharpe" in data and "max_dd" in data:
            results["sharpe_only"] = tur_sharpe_only_null(
                data["observed_house_score"], data["sharpe"], data["max_dd"]
            )

    # Apply FDR correction across all tests
    if len(results) > 1:
        p_values = [r.p_value for r in results.values()]
        fdr_results = benjamini_hochberg_correction(p_values, alpha)

        for i, (null_type, result) in enumerate(results.items()):
            result.fdr_corrected_p = p_values[i]
            result.passed = fdr_results[i]

    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NULL TESTS: Domain-Specific Null Hypotheses for E1 Gates")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    # Layer 1: Consensus
    print("\n[LAYER 1: CONSENSUS NULLS]")
    signals = np.random.randn(n, 5)  # 5 signals
    labels = (signals[:, 0] + signals[:, 1] > 0).astype(int)  # Correlated labels

    def accuracy_metric(sigs, labs):
        predictions = (sigs[:, 0] + sigs[:, 1] > 0).astype(int)
        return np.mean(predictions == labs)

    data_consensus = {
        "signals": signals,
        "labels": labels,
        "metric_fn": accuracy_metric,
        "observed_accuracy": accuracy_metric(signals, labels)
    }

    consensus_results = run_all_nulls_for_layer("consensus", data_consensus)
    for null_type, result in consensus_results.items():
        print(f"  {null_type}: p={result.p_value:.3f}, passed={result.passed}")

    # Layer 2: χ-crash
    print("\n[LAYER 2: CHI-CRASH NULLS]")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

    def chi_fn(p):
        returns = np.diff(np.log(p))
        flux = np.var(returns[-20:])
        dissipation = np.var(returns[-100:])
        return flux / (dissipation + 1e-10)

    data_chi = {
        "prices": prices,
        "chi_fn": chi_fn
    }

    chi_results = run_all_nulls_for_layer("chi_crash", data_chi)
    for null_type, result in chi_results.items():
        print(f"  {null_type}: p={result.p_value:.3f}, passed={result.passed}")

    # Layer 3: Fraud
    print("\n[LAYER 3: FRAUD NULLS]")
    coupling_matrix = np.random.rand(10, 10) * 0.5
    coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2  # Symmetrize

    def s_star_fn(matrix):
        return np.mean(matrix[np.triu_indices(len(matrix), k=1)])

    data_fraud = {
        "coupling_matrix": coupling_matrix,
        "s_star_fn": s_star_fn
    }

    fraud_results = run_all_nulls_for_layer("fraud", data_fraud)
    for null_type, result in fraud_results.items():
        print(f"  {null_type}: p={result.p_value:.3f}, passed={result.passed}")

    # Layer 4: TUR
    print("\n[LAYER 4: TUR NULLS]")
    portfolio_returns = np.random.randn(252) * 0.01 + 0.0005
    observed_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

    data_tur = {
        "prices": prices,
        "portfolio_returns": portfolio_returns,
        "observed_sharpe": observed_sharpe,
        "observed_house_score": 65.0,
        "sharpe": observed_sharpe,
        "max_dd": -0.10
    }

    tur_results = run_all_nulls_for_layer("tur", data_tur)
    for null_type, result in tur_results.items():
        print(f"  {null_type}: p={result.p_value:.3f}, passed={result.passed}")

    print("\n" + "=" * 70)
    print("All null tests completed. Goal: Each layer beats its own nulls.")
    print("=" * 70)
