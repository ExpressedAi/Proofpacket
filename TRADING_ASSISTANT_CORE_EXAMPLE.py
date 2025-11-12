"""
φ-Vortex Trading Assistant: Core Phase-Lock Detection Example

This is a minimal, self-contained example showing how to:
1. Fetch market data
2. Detect phase-locks between two symbols
3. Calculate χ-criticality
4. Find Fibonacci triads

Based on the φ-Vortex framework from UniversalFramework/

Author: φ-Vortex Research Team
Date: 2025-11-12
"""

import numpy as np
from scipy import signal
from scipy.stats import linregress
from typing import Optional, List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta

# =============================================================================
# CONSTANTS (from φ-Vortex framework)
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
CHI_EQ = 1 / (1 + PHI)      # Optimal criticality ≈ 0.381966
ALPHA = 1 / PHI             # Hierarchy constant ≈ 0.618034

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# =============================================================================
# 1. PHASE-LOCK DETECTION
# =============================================================================

def detect_phase_lock(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    max_ratio: int = 10,
    min_K: float = 0.5
) -> Optional[Dict]:
    """
    Detect phase-lock between two price series using Hilbert transform.

    Based on φ-Vortex framework:
    - Calculate instantaneous phase via Hilbert transform
    - Test all m:n ratios up to max_ratio
    - Coupling strength K ∝ 1/(m*n) (A4 axiom)
    - Return strongest lock that passes E4 persistence test

    Args:
        prices_a: Price series for symbol A (numpy array)
        prices_b: Price series for symbol B (numpy array)
        max_ratio: Maximum m or n to test (default: 10)
        min_K: Minimum coupling strength to report (default: 0.5)

    Returns:
        Dictionary with lock details, or None if no lock found
    """

    # 1. Convert to returns (make stationary)
    returns_a = np.diff(np.log(prices_a))
    returns_b = np.diff(np.log(prices_b))

    # 2. Bandpass filter (focus on 5-20 day cycles)
    # This isolates the relevant frequency range for phase-locking
    sos = signal.butter(4, [1/20, 1/5], btype='band', fs=1.0, output='sos')
    filtered_a = signal.sosfilt(sos, returns_a)
    filtered_b = signal.sosfilt(sos, returns_b)

    # 3. Hilbert transform to get instantaneous phase
    analytic_a = signal.hilbert(filtered_a)
    analytic_b = signal.hilbert(filtered_b)
    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)

    # 4. Test all m:n ratios
    best_lock = None
    best_K = 0

    for m in range(1, max_ratio + 1):
        for n in range(1, max_ratio + 1):
            if m == n and m == 1:
                continue  # Skip 1:1 (trivial)

            # Phase difference for m:n lock: m·φ_a - n·φ_b
            phase_diff = m * phase_a - n * phase_b

            # Order parameter (0 = no lock, 1 = perfect lock)
            # This is the "synchronization index"
            order_param = np.abs(np.mean(np.exp(1j * phase_diff)))

            # Coupling strength (A4 axiom: K ∝ 1/(m*n))
            K_theoretical = 1.0 / (m * n)
            K_measured = order_param * K_theoretical

            # E4 persistence test: Split into 2 halves, check both
            # This ensures the lock is stable, not just a transient fluctuation
            mid = len(phase_diff) // 2
            K_half1 = np.abs(np.mean(np.exp(1j * phase_diff[:mid]))) * K_theoretical
            K_half2 = np.abs(np.mean(np.exp(1j * phase_diff[mid:]))) * K_theoretical

            # Persistent if both halves have K > 0.5 * K_measured
            is_persistent = (K_half1 > 0.5 * K_measured) and (K_half2 > 0.5 * K_measured)

            if is_persistent and K_measured > best_K and K_measured >= min_K:
                best_K = K_measured
                best_lock = {
                    "ratio_m": int(m),
                    "ratio_n": int(n),
                    "ratio_str": f"{m}:{n}",
                    "coupling_strength": float(K_measured),
                    "order_parameter": float(order_param),
                    "is_fibonacci": is_fibonacci_ratio(m, n),
                    "phase_coherence": float(1 - np.std(phase_diff) / np.pi),
                    "K_theoretical": float(K_theoretical)
                }

    return best_lock


def is_fibonacci_ratio(m: int, n: int) -> bool:
    """Check if m:n is a Fibonacci ratio (both m and n in Fibonacci sequence)"""
    return (m in FIBONACCI) and (n in FIBONACCI)


# =============================================================================
# 2. χ-CRITICALITY CALCULATION
# =============================================================================

def calculate_chi(prices: np.ndarray, window: int = 30) -> Dict:
    """
    Calculate χ (chi) criticality: χ = flux / dissipation

    For financial markets:
    - Flux = volatility (price fluctuation energy)
    - Dissipation = mean reversion (how fast prices return to trend)

    χ < 1 → stable (mean-reverting)
    χ ≈ 0.382 = 1/(1+φ) → optimal (healthy market)
    χ > 1 → unstable (trending/bubble)

    Args:
        prices: Price series (numpy array)
        window: Rolling window in days (default: 30)

    Returns:
        Dictionary with χ, flux, dissipation, and status
    """

    if len(prices) < window:
        raise ValueError(f"Need at least {window} data points, got {len(prices)}")

    # Use only last 'window' days
    returns = np.diff(np.log(prices[-window:]))

    # Flux: Realized volatility (std of returns, annualized)
    flux = np.std(returns) * np.sqrt(252)

    # Dissipation: Half-life of mean reversion
    # Fit AR(1) model: r_t = α + β*r_{t-1} + ε
    try:
        slope, intercept, r_value, p_value, std_err = linregress(returns[:-1], returns[1:])

        if slope < 0:  # Mean-reverting
            # Half-life = -log(2) / log(β)
            half_life = -np.log(2) / np.log(abs(slope)) if slope != 0 else np.inf
            dissipation = 1.0 / half_life if half_life > 0 and half_life < 1000 else 0.001
        else:  # Trending (no mean reversion)
            dissipation = 0.001  # Very low dissipation
    except Exception:
        dissipation = 0.001

    # Calculate χ
    chi = flux / dissipation if dissipation > 0 else 10.0

    # Determine status
    if chi < CHI_EQ * 0.8:
        status = "stable"
    elif chi < CHI_EQ * 1.5:
        status = "optimal"
    elif chi < 1.0:
        status = "elevated"
    else:
        status = "critical"

    return {
        "chi": float(chi),
        "flux": float(flux),
        "dissipation": float(dissipation),
        "status": status,
        "optimal_chi": float(CHI_EQ),
        "deviation_pct": float((chi - CHI_EQ) / CHI_EQ * 100)
    }


# =============================================================================
# 3. FIBONACCI TRIAD DETECTION
# =============================================================================

def find_fibonacci_triads(
    symbols: List[str],
    prices_dict: Dict[str, np.ndarray],
    min_coupling: float = 0.3
) -> List[Dict]:
    """
    Find 3-symbol triads where all pairwise ratios are Fibonacci.

    Example: AAPL:GOOGL:META = 3:5:8
    - AAPL:GOOGL = 3:5 (Fibonacci)
    - GOOGL:META = 5:8 (Fibonacci)
    - AAPL:META = 3:8 (Fibonacci)

    Args:
        symbols: List of symbol strings
        prices_dict: Dictionary mapping symbol → price array
        min_coupling: Minimum triad coupling strength

    Returns:
        List of triad dictionaries, sorted by coupling strength
    """

    triads = []
    n = len(symbols)

    print(f"Testing {n * (n-1) * (n-2) // 6} possible triads...")

    # Test all triplets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                sym_a, sym_b, sym_c = symbols[i], symbols[j], symbols[k]

                # Ensure all have data
                if not (sym_a in prices_dict and sym_b in prices_dict and sym_c in prices_dict):
                    continue

                # Detect pairwise locks
                lock_ab = detect_phase_lock(prices_dict[sym_a], prices_dict[sym_b], min_K=0.3)
                lock_bc = detect_phase_lock(prices_dict[sym_b], prices_dict[sym_c], min_K=0.3)
                lock_ac = detect_phase_lock(prices_dict[sym_a], prices_dict[sym_c], min_K=0.3)

                if not (lock_ab and lock_bc and lock_ac):
                    continue

                # All must be Fibonacci
                if not (lock_ab["is_fibonacci"] and lock_bc["is_fibonacci"] and lock_ac["is_fibonacci"]):
                    continue

                # Check transitivity: if A:B = m1:n1 and B:C = m2:n2,
                # then A:C should be approximately m1:n2
                expected_m_ac = lock_ab["ratio_m"]
                expected_n_ac = lock_bc["ratio_n"]

                # Allow ±1 error due to noisy data
                if abs(lock_ac["ratio_m"] - expected_m_ac) <= 1 and \
                   abs(lock_ac["ratio_n"] - expected_n_ac) <= 1:

                    # Triad coupling strength (product of pairwise strengths)
                    K_triad = (lock_ab["coupling_strength"] *
                               lock_bc["coupling_strength"] *
                               lock_ac["coupling_strength"])

                    if K_triad >= min_coupling:
                        triads.append({
                            "symbols": [sym_a, sym_b, sym_c],
                            "ratio_ab": lock_ab["ratio_str"],
                            "ratio_bc": lock_bc["ratio_str"],
                            "ratio_ac": lock_ac["ratio_str"],
                            "coupling_strength": float(K_triad),
                            "K_ab": lock_ab["coupling_strength"],
                            "K_bc": lock_bc["coupling_strength"],
                            "K_ac": lock_ac["coupling_strength"],
                            "fibonacci_sequence": get_fibonacci_sequence([
                                lock_ab["ratio_m"], lock_ab["ratio_n"], lock_bc["ratio_n"]
                            ])
                        })

    # Sort by coupling strength (strongest first)
    return sorted(triads, key=lambda x: x["coupling_strength"], reverse=True)


def get_fibonacci_sequence(ratios: List[int]) -> Optional[List[int]]:
    """Check if ratios form consecutive Fibonacci numbers"""
    # e.g., [3, 5, 8] → F_4, F_5, F_6
    for i in range(len(FIBONACCI) - len(ratios) + 1):
        if ratios == FIBONACCI[i:i+len(ratios)]:
            return ratios
    return None


# =============================================================================
# 4. MARKET DATA FETCHING
# =============================================================================

def fetch_prices(symbol: str, days: int = 365) -> np.ndarray:
    """
    Fetch historical prices using yfinance.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        days: Number of days of history

    Returns:
        Numpy array of closing prices
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    return df['Close'].values


# =============================================================================
# 5. EXAMPLE USAGE
# =============================================================================

def main():
    """
    Example workflow:
    1. Fetch data for multiple symbols
    2. Detect pairwise phase-locks
    3. Calculate χ-criticality for each
    4. Find Fibonacci triads
    """

    print("=" * 80)
    print("φ-VORTEX TRADING ASSISTANT - Core Algorithm Demo")
    print("=" * 80)
    print()

    # Define symbols to analyze
    symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"]
    print(f"Analyzing symbols: {', '.join(symbols)}")
    print()

    # 1. Fetch historical data
    print("Step 1: Fetching historical data (365 days)...")
    prices_dict = {}
    for symbol in symbols:
        try:
            prices = fetch_prices(symbol, days=365)
            prices_dict[symbol] = prices
            print(f"  ✓ {symbol}: {len(prices)} days")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    print()

    # 2. Detect pairwise phase-locks
    print("Step 2: Detecting phase-locks between all pairs...")
    locks = []
    for i, sym_a in enumerate(symbols):
        for j, sym_b in enumerate(symbols):
            if i >= j or sym_a not in prices_dict or sym_b not in prices_dict:
                continue

            lock = detect_phase_lock(prices_dict[sym_a], prices_dict[sym_b], min_K=0.5)
            if lock:
                locks.append({
                    "symbol_a": sym_a,
                    "symbol_b": sym_b,
                    **lock
                })
                fib_marker = "🟡 FIBONACCI" if lock["is_fibonacci"] else ""
                print(f"  ✓ {sym_a}:{sym_b} = {lock['ratio_str']} "
                      f"(K={lock['coupling_strength']:.3f}) {fib_marker}")

    print(f"\nFound {len(locks)} phase-locks total")
    fibonacci_locks = [l for l in locks if l["is_fibonacci"]]
    print(f"  → {len(fibonacci_locks)} are Fibonacci ratios ({len(fibonacci_locks)/len(locks)*100:.1f}%)")
    print()

    # 3. Calculate χ-criticality for each symbol
    print("Step 3: Calculating χ-criticality...")
    print(f"  (Optimal: χ = {CHI_EQ:.3f})")
    print()
    for symbol in symbols:
        if symbol not in prices_dict:
            continue

        chi_result = calculate_chi(prices_dict[symbol], window=30)
        status_emoji = {
            "stable": "🟢",
            "optimal": "🟢",
            "elevated": "🟡",
            "critical": "🔴"
        }[chi_result["status"]]

        print(f"  {status_emoji} {symbol}: χ = {chi_result['chi']:.3f} "
              f"({chi_result['status']}, {chi_result['deviation_pct']:+.1f}% from optimal)")

    print()

    # 4. Find Fibonacci triads
    print("Step 4: Finding Fibonacci triads...")
    triads = find_fibonacci_triads(symbols, prices_dict, min_coupling=0.1)

    if triads:
        print(f"Found {len(triads)} Fibonacci triads:")
        print()
        for idx, triad in enumerate(triads[:3], 1):  # Show top 3
            symbols_str = " : ".join(triad["symbols"])
            print(f"  {idx}. {symbols_str}")
            print(f"     Ratios: {triad['ratio_ab']}, {triad['ratio_bc']}, {triad['ratio_ac']}")
            print(f"     Coupling: K = {triad['coupling_strength']:.4f}")
            if triad["fibonacci_sequence"]:
                print(f"     Fibonacci sequence: {triad['fibonacci_sequence']}")
            print()
    else:
        print("  No triads found with current parameters")
        print()

    # 5. Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total phase-locks detected:  {len(locks)}")
    print(f"Fibonacci locks:             {len(fibonacci_locks)} ({len(fibonacci_locks)/len(locks)*100:.1f}%)")
    print(f"Fibonacci triads:            {len(triads)}")
    print()

    # Show strongest lock
    if locks:
        strongest = max(locks, key=lambda x: x["coupling_strength"])
        print(f"Strongest lock:  {strongest['symbol_a']}:{strongest['symbol_b']} = {strongest['ratio_str']}")
        print(f"                 K = {strongest['coupling_strength']:.3f}")
        print(f"                 Fibonacci: {'Yes' if strongest['is_fibonacci'] else 'No'}")
        print()

    # Show most critical symbol (highest χ)
    chi_values = {}
    for symbol in symbols:
        if symbol in prices_dict:
            chi_values[symbol] = calculate_chi(prices_dict[symbol])["chi"]

    if chi_values:
        most_critical = max(chi_values, key=chi_values.get)
        print(f"Most critical symbol:  {most_critical} (χ = {chi_values[most_critical]:.3f})")
        print(f"Most stable symbol:    {min(chi_values, key=chi_values.get)} (χ = {chi_values[min(chi_values, key=chi_values.get)]:.3f})")

    print()
    print("=" * 80)
    print("✓ Analysis complete!")
    print()
    print("Next steps:")
    print("  1. Deploy as FastAPI backend (see TRADING_ASSISTANT_ARCHITECTURE.md)")
    print("  2. Integrate with OpenRouter for chat interface")
    print("  3. Build React frontend with visualizations")
    print("  4. Add real-time WebSocket streaming")
    print("  5. Implement backtesting engine")
    print("=" * 80)


if __name__ == "__main__":
    main()
