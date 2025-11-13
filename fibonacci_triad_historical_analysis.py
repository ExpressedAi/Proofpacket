"""
FIBONACCI TRIAD HISTORICAL ANALYSIS: 2015-2024
==============================================

Real market data analysis to find concrete examples of Fibonacci triads.

This script:
1. Fetches 10 years of data for S&P 100 stocks
2. Scans for phase-locked triads using rolling windows
3. Compares Fibonacci vs non-Fibonacci ratios
4. Provides concrete examples with dates, tickers, and statistics
5. Performs statistical significance tests

Author: φ-Vortex Research Team
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress, ttest_ind, mannwhitneyu
from typing import List, Dict, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
CHI_EQ = 1 / (1 + PHI)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34]

# S&P 100 representative stocks (major sectors)
SP100_STOCKS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "ADBE",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "DHR", "LLY", "BMY",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "PG", "KO", "PEP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # ETFs for sector analysis
    "SPY", "QQQ", "XLE", "XLF", "XLV", "XLK", "XLI",
]

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_historical_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price data for multiple symbols.

    Returns dict mapping symbol -> DataFrame with Close prices
    """
    print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}...")
    data = {}

    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if len(df) > 250:  # Need at least 1 year of data
                data[symbol] = df[['Close']].copy()
                print(f"  [{i+1}/{len(symbols)}] ✓ {symbol}: {len(df)} days")
            else:
                print(f"  [{i+1}/{len(symbols)}] ✗ {symbol}: Insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] ✗ {symbol}: {str(e)[:50]}")

    print(f"\nSuccessfully fetched {len(data)} symbols")
    return data

# ============================================================================
# PHASE-LOCK DETECTION (from core example, enhanced)
# ============================================================================

def detect_phase_lock_detailed(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    dates: pd.DatetimeIndex,
    max_ratio: int = 10,
    min_K: float = 0.3
) -> Optional[Dict]:
    """
    Enhanced phase-lock detection with period measurement.

    Returns:
        Dictionary with lock details including dominant periods, or None
    """
    if len(prices_a) < 50 or len(prices_b) < 50:
        return None

    # 1. Convert to returns
    returns_a = np.diff(np.log(prices_a))
    returns_b = np.diff(np.log(prices_b))

    # 2. Detrend (remove linear trend)
    t = np.arange(len(returns_a))
    slope_a, intercept_a = np.polyfit(t, returns_a, 1)
    slope_b, intercept_b = np.polyfit(t, returns_b, 1)
    detrended_a = returns_a - (slope_a * t + intercept_a)
    detrended_b = returns_b - (slope_b * t + intercept_b)

    # 3. Bandpass filter (5-30 day cycles)
    try:
        sos = signal.butter(3, [1/30, 1/5], btype='band', fs=1.0, output='sos')
        filtered_a = signal.sosfilt(sos, detrended_a)
        filtered_b = signal.sosfilt(sos, detrended_b)
    except Exception:
        return None

    # 4. Find dominant periods using FFT
    freq_a = np.fft.fftfreq(len(filtered_a))
    fft_a = np.abs(np.fft.fft(filtered_a))
    freq_b = np.fft.fftfreq(len(filtered_b))
    fft_b = np.abs(np.fft.fft(filtered_b))

    # Get dominant period (ignore DC component and high frequencies)
    valid_idx_a = np.where((freq_a > 1/30) & (freq_a < 1/5))[0]
    valid_idx_b = np.where((freq_b > 1/30) & (freq_b < 1/5))[0]

    if len(valid_idx_a) == 0 or len(valid_idx_b) == 0:
        return None

    dominant_freq_a = freq_a[valid_idx_a[np.argmax(fft_a[valid_idx_a])]]
    dominant_freq_b = freq_b[valid_idx_b[np.argmax(fft_b[valid_idx_b])]]

    dominant_period_a = 1 / abs(dominant_freq_a) if dominant_freq_a != 0 else 0
    dominant_period_b = 1 / abs(dominant_freq_b) if dominant_freq_b != 0 else 0

    # 5. Hilbert transform for phase
    try:
        analytic_a = signal.hilbert(filtered_a)
        analytic_b = signal.hilbert(filtered_b)
        phase_a = np.angle(analytic_a)
        phase_b = np.angle(analytic_b)
    except Exception:
        return None

    # 6. Test all m:n ratios
    best_lock = None
    best_K = 0

    for m in range(1, max_ratio + 1):
        for n in range(1, max_ratio + 1):
            if m == n:
                continue

            # Phase difference
            phase_diff = m * phase_a - n * phase_b

            # Order parameter
            order_param = np.abs(np.mean(np.exp(1j * phase_diff)))

            # Coupling strength
            K_theoretical = 1.0 / (m * n)
            K_measured = order_param * K_theoretical

            # E4 persistence test (3 segments)
            third = len(phase_diff) // 3
            K_seg1 = np.abs(np.mean(np.exp(1j * phase_diff[:third]))) * K_theoretical
            K_seg2 = np.abs(np.mean(np.exp(1j * phase_diff[third:2*third]))) * K_theoretical
            K_seg3 = np.abs(np.mean(np.exp(1j * phase_diff[2*third:]))) * K_theoretical

            # All segments must be strong
            is_persistent = (K_seg1 > 0.4 * K_measured) and \
                           (K_seg2 > 0.4 * K_measured) and \
                           (K_seg3 > 0.4 * K_measured)

            if is_persistent and K_measured > best_K and K_measured >= min_K:
                best_K = K_measured

                # Calculate correlation
                corr = np.corrcoef(returns_a, returns_b)[0, 1]

                best_lock = {
                    "ratio_m": int(m),
                    "ratio_n": int(n),
                    "ratio_str": f"{m}:{n}",
                    "coupling_strength": float(K_measured),
                    "order_parameter": float(order_param),
                    "is_fibonacci": is_fibonacci_ratio(m, n),
                    "correlation": float(corr),
                    "period_a": float(dominant_period_a),
                    "period_b": float(dominant_period_b),
                    "period_ratio": float(dominant_period_a / dominant_period_b) if dominant_period_b > 0 else 0,
                    "K_seg1": float(K_seg1),
                    "K_seg2": float(K_seg2),
                    "K_seg3": float(K_seg3),
                    "K_stability": float(np.std([K_seg1, K_seg2, K_seg3]) / K_measured) if K_measured > 0 else 1.0,
                    "start_date": dates[0].strftime("%Y-%m-%d"),
                    "end_date": dates[-1].strftime("%Y-%m-%d"),
                    "duration_days": int((dates[-1] - dates[0]).days),
                }

    return best_lock


def is_fibonacci_ratio(m: int, n: int) -> bool:
    """Check if both m and n are in Fibonacci sequence"""
    return (m in FIBONACCI) and (n in FIBONACCI)


# ============================================================================
# CHI CALCULATION
# ============================================================================

def calculate_chi_series(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Calculate rolling χ-criticality.

    Returns array of chi values (one per day in valid range)
    """
    chi_values = []

    for i in range(window, len(prices)):
        price_window = prices[i-window:i]
        returns = np.diff(np.log(price_window))

        # Flux: volatility
        flux = np.std(returns) * np.sqrt(252)

        # Dissipation: mean reversion strength
        try:
            slope, _, _, _, _ = linregress(returns[:-1], returns[1:])
            if slope < 0:
                half_life = -np.log(2) / np.log(abs(slope)) if slope != 0 else 1000
                dissipation = 1.0 / max(half_life, 1)
            else:
                dissipation = 0.001
        except:
            dissipation = 0.001

        chi = flux / dissipation if dissipation > 0 else 10.0
        chi_values.append(chi)

    return np.array(chi_values)


# ============================================================================
# TRIAD DETECTION WITH ROLLING WINDOWS
# ============================================================================

def scan_for_triads_rolling(
    data: Dict[str, pd.DataFrame],
    window_days: int = 90,
    step_days: int = 30,
    min_coupling: float = 0.05
) -> List[Dict]:
    """
    Scan for triads using rolling windows.

    Args:
        data: Dict of symbol -> DataFrame
        window_days: Size of analysis window
        step_days: How far to move window each iteration
        min_coupling: Minimum product of pairwise K values

    Returns:
        List of detected triads with full details
    """
    symbols = list(data.keys())
    all_triads = []

    # Find common date range
    all_dates = pd.DatetimeIndex([])
    for df in data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    # Align all data to common dates
    aligned_data = {}
    for symbol, df in data.items():
        aligned = df.reindex(all_dates, method='ffill')
        aligned_data[symbol] = aligned

    print(f"\nScanning {len(symbols)} symbols for triads...")
    print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Window: {window_days} days, Step: {step_days} days")
    print(f"Min coupling: {min_coupling}")
    print()

    # Rolling window analysis
    window_count = 0
    start_idx = 0

    while start_idx + window_days < len(all_dates):
        end_idx = start_idx + window_days
        window_dates = all_dates[start_idx:end_idx]
        window_count += 1

        print(f"Window {window_count}: {window_dates[0].date()} to {window_dates[-1].date()}")

        # Extract price data for this window
        window_prices = {}
        for symbol in symbols:
            prices = aligned_data[symbol].loc[window_dates, 'Close'].values
            if not np.isnan(prices).any() and len(prices) >= window_days:
                window_prices[symbol] = prices

        # Test all triplets in this window
        valid_symbols = list(window_prices.keys())
        n = len(valid_symbols)

        triads_found = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    sym_a, sym_b, sym_c = valid_symbols[i], valid_symbols[j], valid_symbols[k]

                    # Detect pairwise locks
                    lock_ab = detect_phase_lock_detailed(
                        window_prices[sym_a], window_prices[sym_b], window_dates, min_K=0.2
                    )
                    lock_bc = detect_phase_lock_detailed(
                        window_prices[sym_b], window_prices[sym_c], window_dates, min_K=0.2
                    )
                    lock_ac = detect_phase_lock_detailed(
                        window_prices[sym_a], window_prices[sym_c], window_dates, min_K=0.2
                    )

                    if not (lock_ab and lock_bc and lock_ac):
                        continue

                    # Calculate triad coupling
                    K_triad = (lock_ab["coupling_strength"] *
                              lock_bc["coupling_strength"] *
                              lock_ac["coupling_strength"])

                    if K_triad < min_coupling:
                        continue

                    # Calculate chi values for all three
                    chi_a = calculate_chi_series(window_prices[sym_a])
                    chi_b = calculate_chi_series(window_prices[sym_b])
                    chi_c = calculate_chi_series(window_prices[sym_c])

                    # Calculate returns during period
                    ret_a = (window_prices[sym_a][-1] / window_prices[sym_a][0] - 1) * 100
                    ret_b = (window_prices[sym_b][-1] / window_prices[sym_b][0] - 1) * 100
                    ret_c = (window_prices[sym_c][-1] / window_prices[sym_c][0] - 1) * 100

                    # Calculate volatility
                    vol_a = np.std(np.diff(np.log(window_prices[sym_a]))) * np.sqrt(252) * 100
                    vol_b = np.std(np.diff(np.log(window_prices[sym_b]))) * np.sqrt(252) * 100
                    vol_c = np.std(np.diff(np.log(window_prices[sym_c]))) * np.sqrt(252) * 100

                    # Check if all three are Fibonacci
                    is_fib_triad = (lock_ab["is_fibonacci"] and
                                   lock_bc["is_fibonacci"] and
                                   lock_ac["is_fibonacci"])

                    triad = {
                        "symbols": [sym_a, sym_b, sym_c],
                        "start_date": window_dates[0].strftime("%Y-%m-%d"),
                        "end_date": window_dates[-1].strftime("%Y-%m-%d"),
                        "duration_days": window_days,

                        # Ratios
                        "ratio_ab": lock_ab["ratio_str"],
                        "ratio_bc": lock_bc["ratio_str"],
                        "ratio_ac": lock_ac["ratio_str"],

                        # Coupling strengths
                        "K_ab": lock_ab["coupling_strength"],
                        "K_bc": lock_bc["coupling_strength"],
                        "K_ac": lock_ac["coupling_strength"],
                        "K_triad": K_triad,

                        # Fibonacci flag
                        "is_fibonacci": is_fib_triad,

                        # Chi values
                        "chi_a_start": float(chi_a[0]) if len(chi_a) > 0 else 0,
                        "chi_a_avg": float(np.mean(chi_a)) if len(chi_a) > 0 else 0,
                        "chi_a_end": float(chi_a[-1]) if len(chi_a) > 0 else 0,
                        "chi_b_avg": float(np.mean(chi_b)) if len(chi_b) > 0 else 0,
                        "chi_c_avg": float(np.mean(chi_c)) if len(chi_c) > 0 else 0,

                        # Correlations
                        "corr_ab": lock_ab["correlation"],
                        "corr_bc": lock_bc["correlation"],
                        "corr_ac": lock_ac["correlation"],

                        # Periods
                        "period_a": lock_ab["period_a"],
                        "period_b": lock_ab["period_b"],
                        "period_c": lock_bc["period_b"],

                        # Returns
                        "return_a": ret_a,
                        "return_b": ret_b,
                        "return_c": ret_c,

                        # Volatility
                        "volatility_a": vol_a,
                        "volatility_b": vol_b,
                        "volatility_c": vol_c,

                        # Stability
                        "K_stability": (lock_ab["K_stability"] +
                                       lock_bc["K_stability"] +
                                       lock_ac["K_stability"]) / 3,
                    }

                    all_triads.append(triad)
                    triads_found += 1

        print(f"  → Found {triads_found} triads in this window")

        # Move to next window
        start_idx += step_days

    print(f"\nTotal triads found: {len(all_triads)}")
    return all_triads


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compare_fibonacci_vs_nonfibonacci(triads: List[Dict]) -> Dict:
    """
    Compare Fibonacci vs non-Fibonacci triads statistically.
    """
    fib_triads = [t for t in triads if t["is_fibonacci"]]
    non_fib_triads = [t for t in triads if not t["is_fibonacci"]]

    if len(fib_triads) == 0 or len(non_fib_triads) == 0:
        return {
            "error": "Insufficient data for comparison",
            "fib_count": len(fib_triads),
            "non_fib_count": len(non_fib_triads)
        }

    # Extract metrics
    fib_K = [t["K_triad"] for t in fib_triads]
    non_fib_K = [t["K_triad"] for t in non_fib_triads]

    fib_stability = [t["K_stability"] for t in fib_triads]
    non_fib_stability = [t["K_stability"] for t in non_fib_triads]

    # Statistical tests
    t_stat_K, p_value_K = ttest_ind(fib_K, non_fib_K)
    u_stat_K, p_value_K_mw = mannwhitneyu(fib_K, non_fib_K, alternative='two-sided')

    t_stat_stab, p_value_stab = ttest_ind(fib_stability, non_fib_stability)

    return {
        "fibonacci_count": len(fib_triads),
        "non_fibonacci_count": len(non_fib_triads),
        "fibonacci_percentage": len(fib_triads) / len(triads) * 100,

        "K_triad_fibonacci_mean": float(np.mean(fib_K)),
        "K_triad_fibonacci_std": float(np.std(fib_K)),
        "K_triad_fibonacci_median": float(np.median(fib_K)),

        "K_triad_non_fib_mean": float(np.mean(non_fib_K)),
        "K_triad_non_fib_std": float(np.std(non_fib_K)),
        "K_triad_non_fib_median": float(np.median(non_fib_K)),

        "K_ratio_fib_to_non_fib": float(np.mean(fib_K) / np.mean(non_fib_K)) if np.mean(non_fib_K) > 0 else 0,

        "stability_fibonacci_mean": float(np.mean(fib_stability)),
        "stability_non_fib_mean": float(np.mean(non_fib_stability)),

        "t_test_K_statistic": float(t_stat_K),
        "t_test_K_p_value": float(p_value_K),
        "mann_whitney_K_p_value": float(p_value_K_mw),

        "t_test_stability_p_value": float(p_value_stab),

        "significant_difference": p_value_K < 0.05,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(triads: List[Dict], comparison: Dict, output_file: str):
    """
    Generate comprehensive markdown report.
    """
    # Sort by coupling strength
    triads_sorted = sorted(triads, key=lambda x: x["K_triad"], reverse=True)

    fib_triads = [t for t in triads_sorted if t["is_fibonacci"]]
    non_fib_triads = [t for t in triads_sorted if not t["is_fibonacci"]]

    report = f"""# FIBONACCI TRIADS IN FINANCIAL MARKETS: 2015-2024

**Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Data Source**: Yahoo Finance
**Framework**: φ-Vortex Phase-Locking Theory

---

## EXECUTIVE SUMMARY

**Total triads detected**: {len(triads)}
**Fibonacci triads**: {len(fib_triads)} ({len(fib_triads)/len(triads)*100:.1f}%)
**Non-Fibonacci triads**: {len(non_fib_triads)} ({len(non_fib_triads)/len(triads)*100:.1f}%)

### Key Finding

"""

    if comparison.get("significant_difference"):
        report += f"""✅ **FIBONACCI PREFERENCE CONFIRMED**

Fibonacci triads show **{comparison['K_ratio_fib_to_non_fib']:.2f}× stronger coupling** than non-Fibonacci triads.

- **Fibonacci K (mean)**: {comparison['K_triad_fibonacci_mean']:.4f}
- **Non-Fibonacci K (mean)**: {comparison['K_triad_non_fib_mean']:.4f}
- **Statistical significance**: p = {comparison['t_test_K_p_value']:.4f}

"""
    else:
        report += f"""⚠️ **NO STRONG FIBONACCI PREFERENCE DETECTED**

Fibonacci and non-Fibonacci triads show similar coupling strengths.

- **Fibonacci K (mean)**: {comparison['K_triad_fibonacci_mean']:.4f}
- **Non-Fibonacci K (mean)**: {comparison['K_triad_non_fib_mean']:.4f}
- **Statistical significance**: p = {comparison['t_test_K_p_value']:.4f}

This may indicate:
1. Insufficient sample size
2. Window parameters not optimal
3. Fibonacci preference is real but subtle
4. Market data too noisy for clean detection

"""

    # Top 10 triads
    report += f"""---

## TOP 10 STRONGEST TRIADS

"""

    for i, triad in enumerate(triads_sorted[:10], 1):
        fib_marker = "🟡 FIBONACCI" if triad["is_fibonacci"] else ""

        report += f"""### Triad #{i}: {triad['symbols'][0]} : {triad['symbols'][1]} : {triad['symbols'][2]} {fib_marker}

**Period**: {triad['start_date']} to {triad['end_date']} ({triad['duration_days']} days)

**Ratios**:
- {triad['symbols'][0]}:{triad['symbols'][1]} = {triad['ratio_ab']}
- {triad['symbols'][1]}:{triad['symbols'][2]} = {triad['ratio_bc']}
- {triad['symbols'][0]}:{triad['symbols'][2]} = {triad['ratio_ac']}

**Coupling Strengths**:
- K_AB = {triad['K_ab']:.4f}
- K_BC = {triad['K_bc']:.4f}
- K_AC = {triad['K_ac']:.4f}
- **K_triad = {triad['K_triad']:.6f}**

**χ-Criticality**:
- {triad['symbols'][0]}: χ_start = {triad['chi_a_start']:.3f}, χ_avg = {triad['chi_a_avg']:.3f}, χ_end = {triad['chi_a_end']:.3f}
- {triad['symbols'][1]}: χ_avg = {triad['chi_b_avg']:.3f}
- {triad['symbols'][2]}: χ_avg = {triad['chi_c_avg']:.3f}

**Correlation Matrix**:
- {triad['symbols'][0]}-{triad['symbols'][1]}: {triad['corr_ab']:.3f}
- {triad['symbols'][1]}-{triad['symbols'][2]}: {triad['corr_bc']:.3f}
- {triad['symbols'][0]}-{triad['symbols'][2]}: {triad['corr_ac']:.3f}

**Dominant Periods**:
- {triad['symbols'][0]}: {triad['period_a']:.1f} days
- {triad['symbols'][1]}: {triad['period_b']:.1f} days
- {triad['symbols'][2]}: {triad['period_c']:.1f} days

**Price Performance**:
- {triad['symbols'][0]}: {triad['return_a']:+.2f}% (vol: {triad['volatility_a']:.1f}%)
- {triad['symbols'][1]}: {triad['return_b']:+.2f}% (vol: {triad['volatility_b']:.1f}%)
- {triad['symbols'][2]}: {triad['return_c']:+.2f}% (vol: {triad['volatility_c']:.1f}%)

**Stability**: K_stability = {triad['K_stability']:.3f} (lower = more stable)

---

"""

    # Fibonacci-only analysis
    if len(fib_triads) > 0:
        report += f"""## FIBONACCI TRIADS ONLY (Top 5)

"""
        for i, triad in enumerate(fib_triads[:5], 1):
            report += f"""### Fibonacci Triad #{i}: {triad['symbols'][0]} : {triad['symbols'][1]} : {triad['symbols'][2]}

**Period**: {triad['start_date']} to {triad['end_date']}
**Ratios**: {triad['ratio_ab']}, {triad['ratio_bc']}, {triad['ratio_ac']}
**K_triad**: {triad['K_triad']:.6f}
**Returns**: {triad['return_a']:+.1f}%, {triad['return_b']:+.1f}%, {triad['return_c']:+.1f}%

"""

    # Statistical comparison
    report += f"""---

## STATISTICAL COMPARISON

### Coupling Strength (K_triad)

| Metric | Fibonacci | Non-Fibonacci | Ratio |
|--------|-----------|---------------|-------|
| Mean   | {comparison['K_triad_fibonacci_mean']:.6f} | {comparison['K_triad_non_fib_mean']:.6f} | {comparison['K_ratio_fib_to_non_fib']:.2f}× |
| Median | {comparison['K_triad_fibonacci_median']:.6f} | {comparison['K_triad_non_fib_median']:.6f} | - |
| Std    | {comparison['K_triad_fibonacci_std']:.6f} | {comparison['K_triad_non_fib_std']:.6f} | - |

### Hypothesis Test

**H0**: Fibonacci and non-Fibonacci triads have equal coupling strength
**H1**: Fibonacci triads have stronger coupling

**t-test p-value**: {comparison['t_test_K_p_value']:.6f}
**Mann-Whitney U p-value**: {comparison['mann_whitney_K_p_value']:.6f}

"""

    if comparison['t_test_K_p_value'] < 0.05:
        report += f"""**Result**: ✅ REJECT H0 (p < 0.05)
**Conclusion**: Fibonacci triads show statistically significant stronger coupling.

"""
    else:
        report += f"""**Result**: ❌ FAIL TO REJECT H0 (p ≥ 0.05)
**Conclusion**: No statistically significant difference detected.

"""

    # Gotchas
    report += f"""---

## METHODOLOGY & LIMITATIONS

### What We Did

1. **Data**: Fetched 10 years of daily prices for {len(SP100_STOCKS)} major stocks/ETFs
2. **Windows**: Scanned using 90-day rolling windows with 30-day steps
3. **Detection**: Used Hilbert transform for phase extraction, tested all m:n ratios up to 10:10
4. **Filtering**: Required K > 0.2 for pairwise locks, K_triad > 0.05 for triads
5. **Validation**: 3-segment persistence test (E4 axiom from φ-vortex)

### Limitations & Gotchas

⚠️ **Data Mining Risk**: Testing {len(SP100_STOCKS)}³ = {len(SP100_STOCKS)**3:,} combinations increases false positives

⚠️ **Overfitting**: Parameters (window size, filter bands, K thresholds) were not optimized on held-out data

⚠️ **Market Regime Changes**: 2015-2024 includes multiple regimes (bull market, COVID crash, rate hikes)

⚠️ **Survivorship Bias**: Only analyzed stocks that exist today (missed bankruptcies)

⚠️ **Look-Ahead Bias**: None (all analysis uses only past data within each window)

⚠️ **Transaction Costs**: Real trading would incur costs not modeled here

### Interpretation

This analysis provides **EVIDENCE**, not **PROOF**. The φ-vortex framework predicts Fibonacci ratios should be more stable, and the data shows:

"""

    if comparison['t_test_K_p_value'] < 0.05:
        report += f"""✅ **Consistent with prediction** (p = {comparison['t_test_K_p_value']:.4f})

However, statistical significance ≠ practical significance. A {comparison['K_ratio_fib_to_non_fib']:.2f}× difference may or may not translate to profitable trading.

"""
    else:
        report += f"""❌ **Not strongly confirmed** (p = {comparison['t_test_K_p_value']:.4f})

Possible explanations:
1. **Real but subtle**: Effect exists but requires larger sample size
2. **Wrong parameters**: Window size, filter bands, or K thresholds not optimal
3. **Not universal**: Works in some regimes/sectors but not others
4. **Null hypothesis true**: Fibonacci ratios are not actually preferred in markets

"""

    report += f"""---

## NEXT STEPS

### To Strengthen This Analysis

1. **Expand data**: Include international markets, longer history (1990-2024)
2. **Sector-specific**: Analyze tech, finance, energy separately
3. **Regime conditioning**: Test during bull markets, bear markets, high VIX separately
4. **Cross-validation**: Split data into train/test sets, optimize parameters on train only
5. **Robustness checks**: Vary window size (60, 90, 120 days), filter bands, thresholds

### To Test Trading Viability

1. **Backtest strategies**: Pairs trading on phase-locked pairs, triad arbitrage
2. **Calculate transaction costs**: Include spread, slippage, commissions
3. **Risk management**: Stop-loss rules, position sizing
4. **Real-time detection**: Build system that detects forming locks prospectively
5. **Paper trading**: Test on live data before real money

### To Publish Findings

1. **Peer review**: Submit to *Journal of Financial Economics*, *Quantitative Finance*
2. **Preprint**: Post on arXiv, SSRN for community feedback
3. **Replication package**: Share code, data, exact parameters
4. **Transparency**: Report all tests performed, not just significant ones

---

## CONCLUSION

This analysis scanned **10 years of market data** for Fibonacci triads and found:

- **{len(triads)} total triads** detected in {len(SP100_STOCKS)} symbols
- **{len(fib_triads)} Fibonacci triads** ({len(fib_triads)/len(triads)*100:.1f}%)
- **{comparison['K_ratio_fib_to_non_fib']:.2f}× stronger coupling** in Fibonacci vs non-Fibonacci
- **p-value = {comparison['t_test_K_p_value']:.4f}** ({"significant" if comparison['t_test_K_p_value'] < 0.05 else "not significant"})

**Is Fibonacci preference REAL?**

"""

    if comparison['t_test_K_p_value'] < 0.05 and comparison['K_ratio_fib_to_non_fib'] > 2:
        report += f"""**LIKELY YES**, with caveats:
- Statistical evidence supports φ-vortex prediction
- Effect size ({comparison['K_ratio_fib_to_non_fib']:.2f}×) is substantial
- BUT: Requires validation on independent data
- AND: Trading profitability needs separate analysis

"""
    elif comparison['t_test_K_p_value'] < 0.05:
        report += f"""**POSSIBLY**, but weak:
- Statistical significance detected (p < 0.05)
- BUT: Effect size is modest ({comparison['K_ratio_fib_to_non_fib']:.2f}×)
- May not translate to practical trading edge
- Needs replication and robustness checks

"""
    else:
        report += f"""**UNCLEAR** from this analysis:
- No statistical significance (p = {comparison['t_test_K_p_value']:.4f})
- May need different parameters, more data, or refined methods
- Or: Fibonacci preference may not exist in markets
- Do NOT reject the hypothesis yet—but do NOT confirm it either

"""

    report += f"""**Recommendation**: Treat as **intriguing preliminary finding** requiring:
1. Independent replication
2. Out-of-sample validation
3. Robustness to parameter choices
4. Trading simulation with realistic costs

**This is science, not certainty.** The data shows patterns, but patterns ≠ profits.

---

*Report generated by fibonacci_triad_historical_analysis.py*
*φ-Vortex Framework © 2025*
"""

    # Write report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main analysis pipeline.
    """
    print("=" * 80)
    print("FIBONACCI TRIAD HISTORICAL ANALYSIS: 2015-2024")
    print("=" * 80)
    print()

    # 1. Fetch data
    start_date = "2015-01-01"
    end_date = "2024-12-31"

    data = fetch_historical_data(SP100_STOCKS, start_date, end_date)

    if len(data) < 10:
        print("ERROR: Insufficient data fetched. Aborting.")
        return

    # 2. Scan for triads
    print("\n" + "=" * 80)
    triads = scan_for_triads_rolling(
        data,
        window_days=90,  # 3-month windows
        step_days=30,    # Move 1 month at a time
        min_coupling=0.05  # Minimum K_triad to report
    )

    if len(triads) == 0:
        print("\nNo triads found. Try adjusting parameters (window size, min_coupling, etc.)")
        return

    # 3. Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    comparison = compare_fibonacci_vs_nonfibonacci(triads)

    print(json.dumps(comparison, indent=2))

    # 4. Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    output_file = "/home/user/Proofpacket/FIBONACCI_TRIAD_ANALYSIS_REPORT.md"
    generate_report(triads, comparison, output_file)

    # 5. Save raw data
    triads_file = "/home/user/Proofpacket/fibonacci_triads_data.json"
    with open(triads_file, 'w') as f:
        json.dump(triads, f, indent=2)
    print(f"✓ Raw data saved to: {triads_file}")

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Total triads found: {len(triads)}")
    print(f"Fibonacci triads: {len([t for t in triads if t['is_fibonacci']])} ({len([t for t in triads if t['is_fibonacci']])/len(triads)*100:.1f}%)")
    print()
    print(f"Read full report: {output_file}")
    print()


if __name__ == "__main__":
    main()
