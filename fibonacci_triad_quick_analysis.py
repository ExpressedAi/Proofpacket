"""
FIBONACCI TRIAD QUICK ANALYSIS: 2022-2024
==========================================

Faster proof-of-concept version focusing on:
- 15 major stocks (tech + finance + energy)
- 2-year period (2022-2024)
- Larger step sizes for rolling windows

This demonstrates the methodology while running in < 10 minutes.

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

# Focused stock selection (3 sectors × 5 stocks each)
STOCKS = [
    # Tech mega-caps
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    # Finance
    "JPM", "BAC", "GS", "MS", "C",
    # Energy + broad market
    "XOM", "CVX", "SPY", "QQQ", "XLE",
]

# ============================================================================
# HELPER FUNCTIONS (from full version)
# ============================================================================

def is_fibonacci_ratio(m: int, n: int) -> bool:
    """Check if both m and n are in Fibonacci sequence"""
    return (m in FIBONACCI) and (n in FIBONACCI)


def detect_phase_lock_detailed(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    dates: pd.DatetimeIndex,
    max_ratio: int = 8,  # Reduced from 10 for speed
    min_K: float = 0.3
) -> Optional[Dict]:
    """
    Enhanced phase-lock detection with period measurement.
    """
    if len(prices_a) < 40 or len(prices_b) < 40:
        return None

    # 1. Convert to returns
    returns_a = np.diff(np.log(prices_a))
    returns_b = np.diff(np.log(prices_b))

    # 2. Detrend
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

    # Get dominant period
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
                    "K_stability": float(np.std([K_seg1, K_seg2, K_seg3]) / K_measured) if K_measured > 0 else 1.0,
                    "start_date": dates[0].strftime("%Y-%m-%d"),
                    "end_date": dates[-1].strftime("%Y-%m-%d"),
                    "duration_days": int((dates[-1] - dates[0]).days),
                }

    return best_lock


def calculate_chi_series(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """Calculate rolling χ-criticality."""
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
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Quick analysis pipeline.
    """
    print("=" * 80)
    print("FIBONACCI TRIAD QUICK ANALYSIS: 2022-2024")
    print("=" * 80)
    print()

    # 1. Fetch data
    start_date = "2022-01-01"
    end_date = "2024-11-12"

    print(f"Fetching data for {len(STOCKS)} symbols from {start_date} to {end_date}...")
    data = {}

    for i, symbol in enumerate(STOCKS):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if len(df) > 100:
                data[symbol] = df[['Close']].copy()
                print(f"  [{i+1}/{len(STOCKS)}] ✓ {symbol}: {len(df)} days")
            else:
                print(f"  [{i+1}/{len(STOCKS)}] ✗ {symbol}: Insufficient data")
        except Exception as e:
            print(f"  [{i+1}/{len(STOCKS)}] ✗ {symbol}: {str(e)[:40]}")

    if len(data) < 5:
        print("\nERROR: Insufficient data. Aborting.")
        return

    print(f"\nSuccessfully fetched {len(data)} symbols")

    # 2. Scan for triads
    print("\n" + "=" * 80)
    print("SCANNING FOR TRIADS")
    print("=" * 80)

    symbols = list(data.keys())
    all_triads = []

    # Find common date range
    all_dates = pd.DatetimeIndex([])
    for df in data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    # Align all data
    aligned_data = {}
    for symbol, df in data.items():
        aligned = df.reindex(all_dates, method='ffill')
        aligned_data[symbol] = aligned

    # Use full period as single window
    window_dates = all_dates
    window_prices = {}
    for symbol in symbols:
        prices = aligned_data[symbol].loc[window_dates, 'Close'].values
        if not np.isnan(prices).any():
            window_prices[symbol] = prices

    print(f"Analyzing {len(window_prices)} symbols over {len(window_dates)} days")
    print(f"Period: {window_dates[0].date()} to {window_dates[-1].date()}")
    print()

    # Test all triplets
    valid_symbols = list(window_prices.keys())
    n = len(valid_symbols)
    total_triplets = n * (n-1) * (n-2) // 6

    print(f"Testing {total_triplets} possible triplets...")
    print()

    tested = 0
    found = 0

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tested += 1
                sym_a, sym_b, sym_c = valid_symbols[i], valid_symbols[j], valid_symbols[k]

                if tested % 50 == 0:
                    print(f"  Progress: {tested}/{total_triplets} ({tested/total_triplets*100:.1f}%) - Found {found} triads")

                # Detect pairwise locks
                lock_ab = detect_phase_lock_detailed(
                    window_prices[sym_a], window_prices[sym_b], window_dates, min_K=0.25
                )
                lock_bc = detect_phase_lock_detailed(
                    window_prices[sym_b], window_prices[sym_c], window_dates, min_K=0.25
                )
                lock_ac = detect_phase_lock_detailed(
                    window_prices[sym_a], window_prices[sym_c], window_dates, min_K=0.25
                )

                if not (lock_ab and lock_bc and lock_ac):
                    continue

                # Calculate triad coupling
                K_triad = (lock_ab["coupling_strength"] *
                          lock_bc["coupling_strength"] *
                          lock_ac["coupling_strength"])

                if K_triad < 0.02:  # Lower threshold for quick analysis
                    continue

                # Calculate chi values
                chi_a = calculate_chi_series(window_prices[sym_a])
                chi_b = calculate_chi_series(window_prices[sym_b])
                chi_c = calculate_chi_series(window_prices[sym_c])

                # Calculate returns
                ret_a = (window_prices[sym_a][-1] / window_prices[sym_a][0] - 1) * 100
                ret_b = (window_prices[sym_b][-1] / window_prices[sym_b][0] - 1) * 100
                ret_c = (window_prices[sym_c][-1] / window_prices[sym_c][0] - 1) * 100

                # Calculate volatility
                vol_a = np.std(np.diff(np.log(window_prices[sym_a]))) * np.sqrt(252) * 100
                vol_b = np.std(np.diff(np.log(window_prices[sym_b]))) * np.sqrt(252) * 100
                vol_c = np.std(np.diff(np.log(window_prices[sym_c]))) * np.sqrt(252) * 100

                # Check if all three ratios are Fibonacci
                is_fib_triad = (lock_ab["is_fibonacci"] and
                               lock_bc["is_fibonacci"] and
                               lock_ac["is_fibonacci"])

                triad = {
                    "symbols": [sym_a, sym_b, sym_c],
                    "start_date": window_dates[0].strftime("%Y-%m-%d"),
                    "end_date": window_dates[-1].strftime("%Y-%m-%d"),
                    "duration_days": len(window_dates),

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
                    "K_stability": lock_ab["K_stability"],
                }

                all_triads.append(triad)
                found += 1

                fib_marker = "🟡 FIBONACCI" if is_fib_triad else ""
                print(f"    ✓ Found: {sym_a}:{sym_b}:{sym_c} (K={K_triad:.6f}) {fib_marker}")

    print(f"\n✓ Scan complete: Found {len(all_triads)} triads")

    # 3. Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    fib_triads = [t for t in all_triads if t["is_fibonacci"]]
    non_fib_triads = [t for t in all_triads if not t["is_fibonacci"]]

    print(f"Fibonacci triads: {len(fib_triads)}")
    print(f"Non-Fibonacci triads: {len(non_fib_triads)}")
    print()

    if len(fib_triads) > 0 and len(non_fib_triads) > 0:
        fib_K = [t["K_triad"] for t in fib_triads]
        non_fib_K = [t["K_triad"] for t in non_fib_triads]

        fib_K_mean = np.mean(fib_K)
        non_fib_K_mean = np.mean(non_fib_K)
        ratio = fib_K_mean / non_fib_K_mean if non_fib_K_mean > 0 else 0

        t_stat, p_value = ttest_ind(fib_K, non_fib_K)

        print(f"Fibonacci K (mean):     {fib_K_mean:.6f}")
        print(f"Non-Fibonacci K (mean): {non_fib_K_mean:.6f}")
        print(f"Ratio:                  {ratio:.2f}×")
        print(f"t-test p-value:         {p_value:.6f}")
        print()

        if p_value < 0.05:
            print("✅ SIGNIFICANT DIFFERENCE (p < 0.05)")
        else:
            print("❌ NO SIGNIFICANT DIFFERENCE (p ≥ 0.05)")

        comparison = {
            "fibonacci_count": len(fib_triads),
            "non_fibonacci_count": len(non_fib_triads),
            "K_triad_fibonacci_mean": fib_K_mean,
            "K_triad_non_fib_mean": non_fib_K_mean,
            "K_ratio_fib_to_non_fib": ratio,
            "t_test_K_p_value": p_value,
            "significant_difference": p_value < 0.05,
        }
    else:
        print("⚠️ Insufficient data for comparison")
        comparison = {
            "error": "Insufficient triads for comparison",
            "fibonacci_count": len(fib_triads),
            "non_fibonacci_count": len(non_fib_triads),
        }

    # 4. Display top triads
    print("\n" + "=" * 80)
    print("TOP 10 TRIADS (sorted by K_triad)")
    print("=" * 80)
    print()

    triads_sorted = sorted(all_triads, key=lambda x: x["K_triad"], reverse=True)

    for i, t in enumerate(triads_sorted[:10], 1):
        fib = "🟡 FIB" if t["is_fibonacci"] else ""
        print(f"{i}. {t['symbols'][0]}:{t['symbols'][1]}:{t['symbols'][2]} {fib}")
        print(f"   Ratios: {t['ratio_ab']}, {t['ratio_bc']}, {t['ratio_ac']}")
        print(f"   K_triad: {t['K_triad']:.6f}")
        print(f"   Returns: {t['return_a']:+.1f}%, {t['return_b']:+.1f}%, {t['return_c']:+.1f}%")
        print()

    # 5. Save results
    output_file = "/home/user/Proofpacket/FIBONACCI_TRIAD_QUICK_REPORT.md"
    triads_file = "/home/user/Proofpacket/fibonacci_triads_quick_data.json"

    # Save JSON
    with open(triads_file, 'w') as f:
        json.dump({
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "symbols": list(symbols),
                "total_triads": len(all_triads),
            },
            "comparison": comparison,
            "triads": all_triads,
        }, f, indent=2)

    print(f"✓ Data saved to: {triads_file}")

    # Generate simple report
    with open(output_file, 'w') as f:
        f.write(f"""# FIBONACCI TRIAD QUICK ANALYSIS: 2022-2024

**Period**: {start_date} to {end_date}
**Symbols analyzed**: {len(symbols)}
**Total triads found**: {len(all_triads)}

## Summary Statistics

- **Fibonacci triads**: {comparison.get('fibonacci_count', 0)} ({comparison.get('fibonacci_count', 0)/len(all_triads)*100 if all_triads else 0:.1f}%)
- **Non-Fibonacci triads**: {comparison.get('non_fibonacci_count', 0)} ({comparison.get('non_fibonacci_count', 0)/len(all_triads)*100 if all_triads else 0:.1f}%)

""")

        if comparison.get('significant_difference'):
            f.write(f"""## ✅ FIBONACCI PREFERENCE DETECTED

- **Fibonacci K (mean)**: {comparison['K_triad_fibonacci_mean']:.6f}
- **Non-Fibonacci K (mean)**: {comparison['K_triad_non_fib_mean']:.6f}
- **Ratio**: {comparison['K_ratio_fib_to_non_fib']:.2f}×
- **p-value**: {comparison['t_test_K_p_value']:.6f}

Fibonacci triads show **{comparison['K_ratio_fib_to_non_fib']:.2f}× stronger coupling** (statistically significant).

""")
        else:
            f.write(f"""## ❌ NO SIGNIFICANT FIBONACCI PREFERENCE

""")
            if 't_test_K_p_value' in comparison:
                f.write(f"""- **p-value**: {comparison['t_test_K_p_value']:.6f} (not significant)

""")

        f.write(f"""## Top 10 Triads

""")
        for i, t in enumerate(triads_sorted[:10], 1):
            fib_marker = "🟡 FIBONACCI" if t["is_fibonacci"] else ""
            f.write(f"""### {i}. {t['symbols'][0]} : {t['symbols'][1]} : {t['symbols'][2]} {fib_marker}

- **Ratios**: {t['ratio_ab']}, {t['ratio_bc']}, {t['ratio_ac']}
- **K_triad**: {t['K_triad']:.6f}
- **Period**: {t['start_date']} to {t['end_date']}
- **Returns**: {t['return_a']:+.1f}%, {t['return_b']:+.1f}%, {t['return_c']:+.1f}%
- **Correlations**: AB={t['corr_ab']:.3f}, BC={t['corr_bc']:.3f}, AC={t['corr_ac']:.3f}

""")

    print(f"✓ Report saved to: {output_file}")

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
