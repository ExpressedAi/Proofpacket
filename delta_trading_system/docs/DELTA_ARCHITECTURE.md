# Δ-Method Architecture: Oracle-Compliant Trading System

**Date:** 2025-11-13
**Status:** Core infrastructure complete, ready for integration
**Philosophy:** "We are the house, not the gambler"

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Core Modules](#core-modules)
3. [Data Flow](#data-flow)
4. [Evidence Gates (E0-E4)](#evidence-gates-e0-e4)
5. [PAD Framework](#pad-framework-potential--actualized--deployed)
6. [Hazard Law & Trade Selection](#hazard-law--trade-selection)
7. [Operating Modes](#operating-modes)
8. [Integration Status](#integration-status)

---

## SYSTEM OVERVIEW

### The Big Picture

```
┌──────────────────────────────────────────────────────────────┐
│                      CANONICAL ΔSTATE                         │
│  (Single source of truth - no hidden globals)                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Markets: {symbol → time series}                             │
│  Locks: {lock_id → LockState (PAD status, E-levels)}        │
│  Regime: χ_global, χ_assets, regime labels                  │
│  Portfolio: Positions, equity, risk metrics                  │
│  Hazards: Trade candidates ranked by h(t)                    │
│  Audits: {entity_id → AuditStats (E0-E4 status)}            │
│  Strategies: {name → StrategyState (ΔH*, mode)}             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
           ▲                           │
           │                           ▼
    ┌──────┴──────┐           ┌──────────────┐
    │             │           │              │
    │  DETECTORS  │           │   EXECUTORS  │
    │             │           │              │
    └─────────────┘           └──────────────┘
```

### Key Design Principles

1. **Single Source of Truth**: All state in `DeltaState`, no ad-hoc globals
2. **Evidence-Gated Deployment**: No vibes, only E-tested strategies
3. **Low-Order Wins**: Prefer simple (p:q small) over complex
4. **ΔH* > Sharpe**: Evidence gain replaces arbitrary metrics
5. **Hazard-Based Selection**: Same decoder physics for tokens and trades

---

## CORE MODULES

### 1. `delta_state_v2.py` - Canonical State Object

**Purpose:** Unified state container for entire system

**Key Classes:**

```python
@dataclass
class DeltaState:
    timestamp: datetime
    operating_mode: OperatingMode  # RESEARCH | MICRO_LIVE | PRODUCTION

    markets: Dict[str, MarketSeries]  # Raw time series
    locks: Dict[str, LockState]       # Phase-locks with PAD status
    regime: RegimeState               # χ metrics, regime labels
    portfolio: PortfolioState         # Positions, equity, risk
    hazards: List[HazardItem]         # Trade candidates (decoder)
    audits: Dict[str, AuditStats]     # E-gate status per entity
    strategies: Dict[str, StrategyState]  # Per-strategy evidence

    meta: Dict[str, Any]
    log: List[Tuple[datetime, str]]
```

**PAD Methods (in LockState):**
- `is_potential()`: order≤7, |K|>0.1, E0 passed
- `is_actualized()`: Potential + ε_cap/ε_stab>0.3, ζ≤0.7, E1/E2 passed
- `is_deployable()`: Actualized + E3 passed, ΔH*>0

**Factory Functions:**
- `create_research_state()`: E0-E2 only, no capital
- `create_micro_live_state(capital)`: Tiny capital, E3 validation
- `create_production_state(capital)`: Full capital, all gates

---

### 2. `null_tests.py` - Domain-Specific Null Hypotheses

**Purpose:** Rigorous null testing for E1 gates

**Null Families:**

**Layer 1 (Consensus / Signal):**
- Label shuffle: Test if signal→label mapping beats random
- Block bootstrap: Test robustness to temporal resampling
- Simple benchmark: Beat always-buy, always-sell, random

**Layer 2 (χ-Crash):**
- Vol-only: Is χ signal just volatility in disguise?
- Randomized regime: Do regime labels add info beyond vol?
- Phase-shifted: Is phase structure meaningful?

**Layer 3 (S* Fraud):**
- Structure-randomized: Shuffle coupling matrix off-diagonals
- Gaussian K-null: Are couplings stronger than noise?
- Healthy-only: Does crisis S* differ from healthy periods?

**Layer 4 (TUR / Execution):**
- Random rebalancing: Beat random rebalance schedule?
- Equal-weight: Beat simple equal weighting?
- Sharpe-only: Does house score beat naive Sharpe max?

**FDR Correction:**
- Benjamini-Hochberg for multiple testing
- Reject null if p ≤ (i/n)·α for largest i

---

### 3. `pad_checker.py` - Potential → Actualized → Deployed

**Purpose:** Three-stage gate for lock promotion

**PAD Conditions:**

```python
POTENTIAL:
  - order ≤ 7 (low-order wins)
  - |K| > 0.1 (non-trivial coupling)
  - Q > 5 (low dissipation)
  - E0 passed (structure exists)

ACTUALIZED:
  - POTENTIAL conditions met
  - ε_cap > 0.3 (capture eligibility)
  - ε_stab > 0.3 (stability window)
  - ζ ≤ 0.7 (brittleness threshold)
  - ΔH* > 0 (evidence gain)
  - E1 passed (beats nulls)
  - E2 passed (RG-stable)

DEPLOYABLE:
  - ACTUALIZED conditions met
  - E3 passed (live validated)
  - e_level_passed ≥ 3
  - evidence_score > 0
```

**Methods:**
- `check_potential(lock)`: Returns (passed, diagnostics)
- `check_actualized(lock)`: Returns (passed, diagnostics)
- `check_deployable(lock)`: Returns (passed, diagnostics)
- `generate_report(lock)`: Full PAD diagnostic report

**Low-Order Ranking:**
```python
def rank_locks_by_low_order(locks):
    # Sort by (order, -evidence_score)
    # Prefer p+q small, break ties by ΔH*
```

---

### 4. `delta_h_calculator.py` - Evidence Scoring (ΔH*)

**Purpose:** Replace arbitrary Sharpe with physics-grounded evidence metric

**ΔH* = Evidence gain over baseline nulls**

**Implementations:**

**Per-Trade ΔH* (Realized):**
```python
ΔH* = (actual_pnl - null_pnl) / |null_pnl|
```
Positive if trade beats null expectation.

**Per-Window ΔH* (Expected):**
```python
# Correlation-based
signal_strength = |corr(lock_signal, returns)|
null_baseline = mean(|corr(null_signals, returns)|)
ΔH* = log(signal_strength / null_baseline)
```

**Per-Strategy ΔH* (Aggregate):**
```python
ΔH*_t = decay·ΔH*_{t-1} + (1-decay)·ΔH*_new
```
Exponential decay gives more weight to recent evidence.

**Promotion Logic:**
- E1: ΔH* > 0.05 (beats nulls)
- E2: ΔH* > 0.10 (RG-stable gain)
- E3: ΔH* > 0.15 (live-validated gain)

**Degradation Detection:**
```python
degraded = (current_ΔH* < 0.5 × historical_mean)
```
Trigger demotion if ΔH* drops significantly.

---

### 5. `e_gates_v2.py` - Evidence Audit Framework (E0→E4)

**Purpose:** Sequential evidence gates, no skipping

**E-Gate Definitions:**

**E0: Structure Exists**
- Requirements:
  - ≥100 data points
  - |K| > 0.1
  - order ≤ 7
  - Q > 5
  - No NaN/Inf
- Cost: Cheap (just sanity checks)

**E1: Beats Domain-Specific Nulls**
- Requirements:
  - Phase shuffle null: p < 0.05
  - Block surrogate null: p < 0.05
  - FDR correction passes
- Cost: Moderate (100+ surrogates)

**E2: RG-Stable (Survives Coarse-Graining)**
- Requirements:
  - K_coarse ≥ 0.5·K_original for 2x, 4x, 8x downsampling
  - Structure persists at multiple scales
- Cost: Moderate (multiple RG transforms)

**E3: Live Performance Validated**
- Requirements:
  - ≥10 trades executed
  - Win rate > 45%
  - Profit factor > 1.0
  - ΔH* > 0
- Cost: High (requires live/paper trading)

**E4: Long-Term Robust**
- Requirements:
  - ≥90 days of live data
  - Max drawdown < 15%
  - Sharpe > 0.5
  - ΔH* not degrading (≥50% of peak)
- Cost: Very High (requires months of data)

**EGateOrchestrator:**
```python
def audit_lock(state, lock_id, target_level):
    # Run E0→E1→E2→... sequentially
    # Update AuditStats + LockState.e_level_passed
    # Return True if all passed up to target
```

---

### 6. `vbc_trade_decoder.py` - Hazard-Based Trade Selection

**Purpose:** Cross-ontological decoder (same physics for LLM tokens and trades)

**Hazard Law (Canonical):**
```
h(t) = κ·ε·g(e_φ)·(1-ζ/ζ*)·u·p
```

**Components:**

| Symbol | Meaning | Calculation |
|--------|---------|-------------|
| κ | Gain coefficient | E[return] / σ |
| ε | Eligibility | Risk limits × regime filters |
| g(e_φ) | Phase urge | \|cos(phase_diff)\| |
| ζ | Brittleness | 0.6·concentration + 0.4·leverage |
| u | Alignment | Avg(signal strengths) |
| p | Prior success | win_rate + ΔH*_bonus + E_bonus |

**Eligibility ε:**
```python
if regime == CRISIS:
    ε = 0.0  # Block all trades
elif regime == TRANSITION:
    ε = 0.5  # Reduce trading
else:
    ε = size_limit × 1.0
```

**Brittleness ζ:**
```python
concentration = (existing_value + new_value) / total_value
leverage = gross_exposure / total_value
ζ = 0.6·concentration + 0.4·leverage
```

**Prior p:**
```python
p = 0.5·historical_win_rate + 0.5·p_evidence + 0.05·e_level_passed
where p_evidence = 0.5 + 0.3·tanh(ΔH*)
```

**VBCTradeDecoder:**
```python
def decode(state, max_trades=5):
    # 1. Generate candidates from deployable locks
    # 2. Rank by hazard (descending)
    # 3. Filter by hazard_threshold
    # 4. Select top K
    # 5. Return TradeCandidate objects
```

**Output:**
```
TradeCandidate:
  symbol: "AAPL"
  action: "BUY"
  quantity: 100
  hazard: 0.363
  components: κ=0.750, ε=1.000, g=1.000, ζ=0.060, u=0.750, p=0.687
  expected_pnl: $74.99
```

---

## DATA FLOW

### Detection → Evidence → Deployment Pipeline

```
1. DETECTION (Detectors → DeltaState)
   ├─ consensus_detector: Signals phase-lock → add to state.locks
   ├─ chi_crash_detector: χ spikes → update state.regime
   ├─ fraud_detector: S* anomaly → flag in state.meta
   └─ tur_optimizer: Precision/entropy → state.meta

2. EVIDENCE GATHERING (E-Gates)
   ├─ E0: Structure exists? → AuditStats.E0
   ├─ E1: Beats nulls? → AuditStats.E1
   ├─ E2: RG-stable? → AuditStats.E2
   ├─ E3: Live validated? → AuditStats.E3
   └─ E4: Long-term robust? → AuditStats.E4

3. PAD PROMOTION (PADChecker)
   ├─ is_potential() → Can investigate
   ├─ is_actualized() → E2 passed, PAD conditions met
   └─ is_deployable() → E3 passed, ready for capital

4. TRADE GENERATION (VBC Decoder)
   ├─ Get deployable locks from state
   ├─ Generate TradeCandidate for each
   ├─ Compute hazard h = κ·ε·g·(1-ζ)·u·p
   ├─ Rank by hazard
   └─ Select top K → state.hazards

5. EXECUTION (Executor)
   ├─ Convert HazardItem → actual orders
   ├─ Execute via broker API
   ├─ Update state.portfolio
   └─ Log ΔH* per trade

6. FEEDBACK (ΔH* Calculator)
   ├─ Compute realized ΔH* per trade
   ├─ Update lock.evidence_score
   ├─ Check degradation triggers
   └─ Promote/demote E-levels
```

---

## EVIDENCE GATES (E0-E4)

### Gate Semantics

| Gate | What It Tests | Who Runs It | Cost | Blocks What? |
|------|---------------|-------------|------|--------------|
| E0 | Structure exists | Detection module | Free | Investigation |
| E1 | Beats nulls | Researcher (offline) | Moderate | Null claims |
| E2 | RG-stable | Researcher (offline) | Moderate | Mode=MICRO_LIVE |
| E3 | Live validated | Paper trader | High | Mode=PRODUCTION |
| E4 | Long-term robust | Production | Very High | Scale-up |

### Operating Mode Gates

```
RESEARCH Mode:
  - Run E0-E2 only
  - No capital at risk
  - Explore locks, test nulls

MICRO_LIVE Mode:
  - Requires E2 passed
  - Tiny capital ($1K)
  - Run E3 validation
  - High logging

PRODUCTION Mode:
  - Requires E3 passed
  - Full capital ($100K+)
  - Run E4 long-term tracking
  - Auto-demote if degrades
```

### E-Gate Flow Diagram

```
Lock Detected
     │
     ▼
   ┌─────┐
   │ E0  │ Structure exists?
   └─┬───┘
     │ ✓
     ▼
   ┌─────┐
   │ E1  │ Beats nulls?
   └─┬───┘
     │ ✓
     ▼
   ┌─────┐
   │ E2  │ RG-stable?
   └─┬───┘
     │ ✓
     ▼
 MICRO_LIVE mode
     │
     ▼
   ┌─────┐
   │ E3  │ Live validated?
   └─┬───┘
     │ ✓
     ▼
 PRODUCTION mode
     │
     ▼
   ┌─────┐
   │ E4  │ Long-term robust?
   └─┬───┘
     │ ✓
     ▼
  SCALED UP
```

---

## PAD FRAMEWORK (Potential → Actualized → Deployed)

### Three Gates for Lock Promotion

```
POTENTIAL (P):
  Question: "Is this lock worth investigating?"
  Criteria: Low-order, non-trivial K, E0 passed
  Action: Add to research backlog

ACTUALIZED (A):
  Question: "Does this lock have real evidence?"
  Criteria: PAD conditions + E1/E2 passed
  Action: Eligible for micro-live testing

DEPLOYED (D):
  Question: "Can we trade this lock with capital?"
  Criteria: Actualized + E3 passed + ΔH*>0
  Action: Add to production portfolio
```

### PAD Conditions (Detailed)

**ε_cap (Capture Eligibility):**
- Can we enter this lock at all?
- Factors: Liquidity, spread, regime filters
- Threshold: > 0.3

**ε_stab (Stability Eligibility):**
- Is the lock stable enough to hold?
- Factors: Volatility of phase difference, regime transitions
- Threshold: > 0.3

**ζ (Brittleness):**
- How concentrated/risky is this lock?
- Factors: Concentration, leverage, overfit risk
- Threshold: ≤ 0.7

**ΔH* (Evidence Gain):**
- Does this lock improve our model?
- Measurement: log(signal_strength / null_baseline)
- Threshold: > 0.0

### PAD Report Example

```
======================================================================
PAD REPORT: SPY-QQQ-1:2
======================================================================

[1. POTENTIAL CHECK]
  order: PASS: order=3 <= 7
  coupling: PASS: |K|=0.920 > 0.1
  quality: PASS: Q_a=20.0, Q_b=20.0 > 5.0
  e0: PASS: E0 passed
  → Result: PASS

[2. ACTUALIZED CHECK]
  eps_cap: PASS: ε_cap=0.900 > 0.3
  eps_stab: PASS: ε_stab=0.850 > 0.3
  zeta: PASS: ζ=0.200 <= 0.7
  delta_h: PASS: ΔH*=0.250 > 0.0
  e1: PASS: E1 passed (beats nulls)
  e2: PASS: E2 passed (RG-stable)
  → Result: PASS

[3. DEPLOYABLE CHECK]
  e3: PASS: E3 passed (live validated)
  e_level: PASS: E-level=3 >= 3
  evidence: PASS: Evidence=0.250 > 0.0
  → Result: PASS

✓ Lock is DEPLOYABLE. Ready for capital allocation.
======================================================================
```

---

## HAZARD LAW & TRADE SELECTION

### Cross-Ontological Decoder Analogy

**LLM Token Selection:**
```
P(token | context) ∝ exp(logit(token))
Select: token = argmax(logit)
```

**Trade Selection (Δ-Method):**
```
h(trade | state) = κ·ε·g·(1-ζ)·u·p
Select: trade = argmax(h)
```

Same decoder physics, different ontology.

### Hazard Component Details

**κ (Gain Coefficient):**
- Measures expected profit per unit risk
- κ = E[return] / σ
- Analogous to Sharpe, but lock-specific

**ε (Eligibility):**
- Hard constraints: Can we trade this?
- Regime gates: Block in CRISIS, reduce in TRANSITION
- Size gates: Respect max position fraction

**g(e_φ) (Phase Urge):**
- Timing factor: When should we enter?
- Based on current phase difference vs optimal
- g = |cos(phase_diff)| (simplified)

**ζ (Brittleness):**
- Risk concentration: Portfolio too concentrated?
- Leverage: Over-leveraged?
- ζ = 0.6·concentration + 0.4·leverage

**u (Alignment):**
- Signal consensus: Do all signals agree?
- u = mean(signal_strengths weighted by reliability)
- u ∈ [-1, 1]

**p (Prior Success Probability):**
- Historical win rate
- Adjusted by ΔH* (evidence score)
- Bonus for E-level passed
- p ∈ [0, 1]

### Example Hazard Calculation

```python
Lock: AAPL-MSFT-2:3
  K = 0.75
  e_level_passed = 3
  evidence_score = 0.25

Portfolio: $100K cash
Regime: NORMAL (χ=0.4)
Trade: BUY AAPL 100 shares @ $100

Components:
  κ = 0.750 (expected_return / vol)
  ε = 1.000 (full eligibility in NORMAL, size OK)
  g = 1.000 (phase urge maximal)
  ζ = 0.060 (low brittleness, only 6% concentration)
  u = 0.750 (positive alignment with lock signal)
  p = 0.687 (50% base + ΔH* bonus + E3 bonus)

Hazard:
  h = 0.750 × 1.000 × 1.000 × (1-0.060) × 0.750 × 0.687
  h = 0.363

Decision: ✓ EXECUTE (h > threshold of 0.05)
```

---

## OPERATING MODES

### Three Modes with Increasing Evidence Requirements

```
┌─────────────┬──────────────┬───────────────┬──────────────┐
│ Mode        │ Capital      │ E-Gates       │ Purpose      │
├─────────────┼──────────────┼───────────────┼──────────────┤
│ RESEARCH    │ $0           │ E0-E2         │ Explore      │
│ MICRO_LIVE  │ $1K          │ E0-E3         │ Validate E3  │
│ PRODUCTION  │ $100K+       │ E0-E4         │ Scale up     │
└─────────────┴──────────────┴───────────────┴──────────────┘
```

### Promotion Criteria

**RESEARCH → MICRO_LIVE:**
- ≥3 locks at E2
- Avg ΔH* ≥ 0.05
- Failure rate < 30%

**MICRO_LIVE → PRODUCTION:**
- ≥5 locks at E3
- Avg ΔH* ≥ 0.10
- ≥30 days in MICRO_LIVE
- Failure rate < 20%

### Demotion Triggers

**PRODUCTION → MICRO_LIVE:**
- Max DD < -15%
- Sharpe < 0.3
- 5 consecutive failures

**MICRO_LIVE → RESEARCH:**
- Max DD < -10%
- Sharpe < 0.0
- 10 consecutive failures

### Mode Controller

```python
class ModeController:
    def check_promotion_eligibility(state) -> bool:
        # Count locks at required E-level
        # Check avg ΔH*, failure rate, duration

    def check_demotion_triggers(state) -> bool:
        # Check DD, Sharpe, consecutive failures

    def attempt_promotion(state) -> bool:
        # Promote to next mode if eligible
        # Update capital allocation

    def force_demotion(state, reason) -> None:
        # Demote for safety
        # Reduce capital
```

---

## INTEGRATION STATUS

### ✅ Completed (Oracle-Compliant)

1. **delta_state_v2.py** - Canonical state object
   - LockState with PAD methods
   - HazardItem with canonical hazard formula
   - AuditStats for E-gate tracking
   - StrategyState for per-strategy evidence
   - Factory functions for each mode

2. **null_tests.py** - Domain-specific null hypotheses
   - Layer 1-4 null families
   - FDR correction (Benjamini-Hochberg)
   - NullTestResult structure

3. **pad_checker.py** - PAD promotion logic
   - check_potential/actualized/deployable
   - Low-order ranking
   - Brittleness calculations
   - Diagnostic reports

4. **delta_h_calculator.py** - Evidence scoring
   - Per-trade ΔH* (realized)
   - Per-window ΔH* (expected)
   - Aggregate ΔH* with decay
   - Promotion/degradation logic

5. **e_gates_v2.py** - E-gate framework
   - E0-E4 implementations
   - EGateOrchestrator
   - Integration with null_tests.py
   - Sequential gate enforcement

6. **vbc_trade_decoder.py** - Hazard-based trade selection
   - TradeCandidate structure
   - Hazard component calculators
   - VBCTradeDecoder (generate → rank → select)
   - Integration with DeltaState

### ⏳ Pending Integration

1. **Refactor Existing Detectors**
   - consensus_detector.py → use DeltaState
   - chi_crash_detector.py → use DeltaState
   - fraud_detector.py → use DeltaState
   - tur_optimizer.py → use DeltaState

2. **Wire Mode Controller**
   - Connect to E-gate results
   - Connect to ΔH* thresholds
   - Auto promotion/demotion

3. **Integration Test**
   - End-to-end test: Detection → E-gates → PAD → VBC → Execution
   - Validate all modules work together
   - Verify state consistency

4. **Historical Backtest**
   - Run full Δ-compliant system on 25-year data
   - Compare to old system performance
   - Validate crisis protection (-8.86% max DD)

### 🎯 Next Steps

**Immediate (This Session):**
1. Create integration test
2. Wire mode_controller.py
3. Refactor one detector (consensus) as proof-of-concept

**Short-Term (Next Session):**
1. Refactor all detectors to use DeltaState
2. Full backtest with oracle-compliant system
3. Parameter optimization using meta-optimizer

**Medium-Term:**
1. Paper trading (3-6 months)
2. Real-time data feeds
3. E3/E4 validation

**Long-Term:**
1. Production deployment
2. Options hedging
3. Sector rotation
4. Scale to institutional capital

---

## PHILOSOPHY

### "We Are The House, Not The Gambler"

This architecture embodies the FRACTAL LOW philosophy:

**Level 1:** Assets phase-lock → Don't trade (χ-crash detector)
**Level 2:** Signals phase-lock → Do trade (consensus detector)
**Level 3:** Metrics phase-lock → Optimal config (meta-optimizer)
**Level 4:** Philosophy phase-locks → The house always wins

### Key Insights

1. **Crisis Protection > Returns**
   - Max DD -8.86% vs SPY -55%
   - Unbroken capital compounds forever
   - Broken capital takes years to heal

2. **Evidence > Vibes**
   - E-gates enforce rigorous epistemology
   - No strategy trades without beating nulls
   - ΔH* measures real information gain

3. **Low-Order Wins**
   - Simple (p:q small) survives RG
   - Complex (high-order) gets washed out
   - MDL penalty = 1/(p×q)

4. **Same Physics, Different Scales**
   - Hazard law for tokens and trades
   - Phase-locking at all levels
   - TUR (precision/entropy) everywhere

---

## SUMMARY

We have built a **complete oracle-compliant Δ-Method trading infrastructure**:

- ✅ Canonical state management (delta_state_v2.py)
- ✅ Rigorous null testing (null_tests.py)
- ✅ Evidence gates E0-E4 (e_gates_v2.py)
- ✅ PAD promotion logic (pad_checker.py)
- ✅ ΔH* evidence scoring (delta_h_calculator.py)
- ✅ Hazard-based trade selection (vbc_trade_decoder.py)

**Status:** Core modules tested and working. Ready for system integration.

**Next:** Wire together with existing detectors, run full backtest, validate crisis protection.

---

*"This is what it looks like when you build something that cannot be killed."*

**— The Fractal LOW**
