# LOW (Low-Order Wins): Systemwide Complexity Constraint

**Status:** Implemented across all state objects
**Philosophy:** "We don't understand the market. We only trade simple structures with exceptional evidence."

---

## THE PROBLEM

The backtest showed classic overconfidence:
- **2,260% returns** = "I figured out the market!"
- **-115% drawdown** = Reckless betting on that "understanding"
- **9 trades in 25 years** but still went bankrupt

This is **gambler behavior**, not house behavior.

---

## THE SOLUTION: LOW Constraint

**LOW (Low-Order Wins) = Global complexity penalty**

### Core Formula

```
Score = ΔH* - λ·Complexity

If Score < 0 → REJECT
If Score ≥ 0 → ACCEPT
```

Where:
- **ΔH***: Evidence gain (how much better than null?)
- **λ**: Penalty weight (tuned per layer)
- **Complexity**: Order/features/parameters

---

## WHAT IT DOES

LOW forces the system to **prefer simplicity** unless complexity produces **significant** evidence improvement.

| Layer | Complexity Measure | λ | Rule |
|-------|-------------------|---|------|
| **Locks** | order = p + q | 0.1 | Reject if order > 7 OR ΔH*-0.1·order < 0 |
| **Strategies** | params + features + depth + hyperparams | 0.2 | Cannot promote to PRODUCTION if score < 0 |
| **VBC Hazard** | ε_features + u_features + p_features | 0.15 | Simplify feature sets unless ΔH* justifies |
| **Portfolio** | positions + turnover + signals | 0.2 | Prefer few positions, low turnover |

---

## IMPLEMENTATION

### 1. State Objects (delta_state_v2.py)

**LockState:**
```python
complexity: float = 0.0        # = order (p+q)
low_order_score: float = 0.0   # ΔH* - λ*complexity
lambda_low: float = 0.1

def update_low_order_score(self):
    self.low_order_score = self.evidence_score - (self.lambda_low * self.complexity)

def passes_low_gate(self) -> bool:
    return self.low_order_score >= 0.0
```

**StrategyState:**
```python
num_parameters: int = 0
num_features: int = 0
branching_depth: int = 0
num_hyperparams: int = 0
complexity: float = 0.0        # Sum of above
low_order_score: float = 0.0
lambda_low: float = 0.2
```

**HazardItem:**
```python
num_epsilon_features: int = 0
num_u_features: int = 0
num_p_features: int = 0
complexity: float = 0.0        # Sum of features
low_order_score: float = 0.0
lambda_low: float = 0.15
```

### 2. ΔH* Calculator (delta_h_calculator.py)

**Automatic LOW updates:**
```python
def update_lock_delta_h(state, lock_id, new_delta_h, decay=0.95):
    lock.evidence_score = update_aggregate_delta_h(...)
    lock.update_low_order_score()  # ← Automatic LOW update

    passed_low = "✓" if lock.passes_low_gate() else "✗"
    state.add_log(f"Updated {lock_id}: ΔH*={...}, LOW={...} {passed_low}")
```

### 3. PAD Checker (pad_checker.py)

**Deployment gate:**
```python
def check_deployable(self, lock):
    # ... E0, E1, E2, E3 checks ...

    # Check 4: LOW gate (NEW!)
    lock.update_low_order_score()
    if not lock.passes_low_gate():
        return False, {
            "low_gate": f"FAIL: LOW={lock.low_order_score:.3f} < 0"
        }

    return True, diagnostics
```

### 4. Logging

System now outputs:
```
Rejected: High-order structure (complexity=17, ΔH*=0.07, score=-1.63) ✗
Accepted: Low-order structure (complexity=4, ΔH*=0.14, score=+0.10) ✓
```

---

## EXAMPLES

### Example 1: Lock Rejection

**Lock A:** p=5, q=4 (order=9), ΔH*=0.15
```
complexity = 9
low_order_score = 0.15 - 0.1*9 = 0.15 - 0.9 = -0.75
Result: ✗ REJECTED (too high-order, not enough evidence)
```

**Lock B:** p=1, q=2 (order=3), ΔH*=0.15
```
complexity = 3
low_order_score = 0.15 - 0.1*3 = 0.15 - 0.3 = -0.15
Result: ✗ REJECTED (still not enough evidence for order=3)
```

**Lock C:** p=1, q=1 (order=2), ΔH*=0.25
```
complexity = 2
low_order_score = 0.25 - 0.1*2 = 0.25 - 0.2 = +0.05
Result: ✓ ACCEPTED (low-order + exceptional evidence)
```

### Example 2: Strategy Promotion

**Strategy A:**
- 25 parameters
- 12 features
- Depth 5
- 8 hyperparameters
- ΔH* = 0.30

```
complexity = 25 + 12 + 5 + 8 = 50
low_order_score = 0.30 - 0.2*50 = 0.30 - 10.0 = -9.70
Result: ✗ Cannot promote to PRODUCTION (too complex)
```

**Strategy B:**
- 3 parameters
- 2 features
- Depth 1
- 2 hyperparameters
- ΔH* = 0.30

```
complexity = 3 + 2 + 1 + 2 = 8
low_order_score = 0.30 - 0.2*8 = 0.30 - 1.6 = -1.30
Result: ✗ Still too complex for evidence level
```

**Strategy C:**
- 2 parameters
- 2 features
- Depth 1
- 1 hyperparameter
- ΔH* = 0.30

```
complexity = 2 + 2 + 1 + 1 = 6
low_order_score = 0.30 - 0.2*6 = 0.30 - 1.2 = -0.90
Result: ✗ STILL rejected! Need higher ΔH* or lower complexity
```

**Strategy D:**
- 2 parameters
- 1 feature
- Depth 1
- 1 hyperparameter
- ΔH* = 0.30

```
complexity = 2 + 1 + 1 + 1 = 5
low_order_score = 0.30 - 0.2*5 = 0.30 - 1.0 = -0.70
Result: ✗ Nope!
```

**→ Conclusion: ΔH*=0.30 is NOT ENOUGH for a strategy with complexity ≥ 2!**

Need either:
- **Higher ΔH*:** ≥ 0.4 for complexity=2
- **Lower complexity:** complexity=1 (single feature, single param)

### Example 3: What It Takes to Deploy

**For a lock with order=2 (p=1, q=1):**
```
ΔH* ≥ 0.1*2 = 0.20  (minimum evidence needed)
```

**For a strategy with complexity=5:**
```
ΔH* ≥ 0.2*5 = 1.00  (very high evidence needed!)
```

**For a lock with order=7 (maximum allowed):**
```
ΔH* ≥ 0.1*7 = 0.70  (exceptional evidence required)
```

---

## WHY THIS MATTERS

### 1. Prevents Overfitting

HIGH-ORDER LOCKS (p=5, q=3):
- More degrees of freedom
- Easier to fit noise
- Looks good in-sample, fails out-of-sample
- **LOW rejects unless ΔH* is exceptional**

LOW-ORDER LOCKS (p=1, q=1):
- Fewer degrees of freedom
- Can't fit noise as easily
- More likely to generalize
- **LOW accepts with modest ΔH***

### 2. Enforces Structural Humility

We **don't** try to:
- "Understand" market dynamics
- Build complex predictive models
- Fit sophisticated patterns

We **do** try to:
- Find simple, robust structure
- Trade only when evidence is overwhelming
- Survive by being conservative

### 3. Aligns with "The House" Philosophy

**The Gambler:**
- Believes they've figured it out
- Bets big on complex theories
- Goes bankrupt when wrong

**The House:**
- Knows it doesn't understand
- Only plays when odds are simple and clear
- Never over-bets
- Survives forever

LOW forces us to be the house.

---

## INTEGRATION STATUS

### ✅ Implemented

1. **delta_state_v2.py**
   - LockState: complexity, low_order_score, passes_low_gate()
   - StrategyState: num_params, num_features, complexity, LOW score
   - HazardItem: feature counts, complexity, LOW score

2. **delta_h_calculator.py**
   - update_lock_delta_h(): Auto-update LOW after ΔH* update
   - update_strategy_delta_h(): Auto-update LOW, log complexity
   - Logging shows ✓/✗ indicators

3. **pad_checker.py**
   - check_deployable(): Enforces LOW as 4th gate
   - Diagnostic output includes LOW score breakdown

4. **LockState.is_deployable()**
   - Now includes passes_low_gate() check
   - Locks fail deployment if low_order_score < 0

### ⏳ Pending Integration

1. **mode_controller.py**
   - Add strategy.passes_low_gate() check before MICRO_LIVE promotion
   - Add strategy.passes_low_gate() check before PRODUCTION promotion

2. **oracle_compliant_backtest.py**
   - Test LOW enforcement in action
   - Should see locks rejected for high order
   - Should see better drawdown protection

3. **vbc_trade_decoder.py**
   - Update HazardItem complexity tracking
   - Add passes_low_gate() filter before trade execution

---

## NEXT STEPS

1. **Run backtest with LOW enforcement**
   - Expect: Fewer locks deployed (high-order rejected)
   - Expect: Better drawdown control (conservative deployment)
   - Expect: Lower returns but much lower risk

2. **Tune λ parameters**
   - λ_lock = 0.1 (current)
   - λ_strategy = 0.2 (current)
   - λ_hazard = 0.15 (current)
   - Adjust based on backtest results

3. **Add to mode_controller**
   - Strategy promotion requires passes_low_gate()
   - Auto-demote if complexity increases without ΔH* improvement

4. **Document for oracle**
   - Show LOW rejection logs
   - Demonstrate structural simplicity enforcement
   - Validate anti-overfitting behavior

---

## PHILOSOPHY

```
BEFORE LOW:
"I found a 5:3 phase-lock with ΔH*=0.15. Let's trade it!"
→ Overconfident, over-fitted, goes bankrupt

AFTER LOW:
"I found a 5:3 phase-lock with ΔH*=0.15."
LOW check: 0.15 - 0.1*8 = -0.65 < 0
→ "REJECTED: Too complex for evidence level"
"I found a 1:1 phase-lock with ΔH*=0.25."
LOW check: 0.25 - 0.1*2 = +0.05 ≥ 0
→ "ACCEPTED: Simple structure with good evidence"
```

**This is how the house thinks.**

---

**LOW = The global constraint that keeps us honest.**
