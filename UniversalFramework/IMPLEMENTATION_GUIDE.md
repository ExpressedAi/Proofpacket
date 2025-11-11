# VBC Implementation Guide

**For Developers Building Phase-Locked Systems**

This guide shows you how to integrate the Variable Barrier Controller into your AI, robotics, or decision-making systems.

---

## Quick Start (5 Minutes)

### Installation

```bash
# Clone repository
git clone https://github.com/YourUsername/UniversalFramework.git
cd UniversalFramework

# Install dependencies (minimal)
pip install numpy

# Run tests
python vbc_prototype.py
```

### Basic Usage

```python
from vbc_prototype import VariableBarrierController
import numpy as np

# Create VBC with threshold h* = 0.5
vbc = VariableBarrierController(h_star=0.5, top_k=10)

# Your LLM gives you logits
logits = model.get_logits(prompt)

# VBC decides when to commit
result = vbc.process_logits(logits)

if result is not None:
    # Committed to token!
    print(f"Selected token: {result.token_id}")
    print(f"Hazard score: {result.hazard:.3f}")
else:
    # Still deliberating, call again
    pass
```

---

## Core Concepts

### The Hazard Function

Every decision is governed by:

```python
h = κ · ε · g(e_φ) · (1 - ζ/ζ*) · u · p
```

Where:
- **κ**: Sensitivity calibration (usually 1.0)
- **ε**: Capture window (eligibility to commit)
- **g**: Phase coherence (timing fit)
- **ζ**: Brittleness (effort cost)
- **ζ***: Brittleness threshold (budget limit)
- **u**: Semantic alignment (context fit)
- **p**: Prior probability (base rate)

**When h > h***, the system commits to that option.

### The Tick Cycle

VBC operates in 4 phases:

1. **CAPTURE**: Gather top-k candidates from logits
2. **CLEAN**: Filter by alignment threshold
3. **BRIDGE**: Compute ε, g, ζ for each candidate
4. **COMMIT**: If max(h) > h*, emit token

Each call to `process_logits()` advances one phase.

---

## Integration Patterns

### Pattern 1: LLM Token Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vbc_prototype import VariableBarrierController

class VBC_GPT2:
    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vbc = VariableBarrierController(h_star=0.5, top_k=50)

    def generate(self, prompt, max_tokens=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated = []

        for _ in range(max_tokens):
            # Get logits from model
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :].detach().numpy()

            # VBC deliberation
            result = None
            for tick in range(12):  # Max 3 cycles
                result = self.vbc.process_logits(logits,
                                                 token_strings=[self.tokenizer.decode([i])
                                                               for i in range(len(logits))])
                if result is not None:
                    break

            if result is None:
                # Fallback: greedy
                token_id = logits.argmax()
            else:
                token_id = result.token_id

            # Append token
            generated.append(token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[token_id]])], dim=1)

            # Stop tokens
            if token_id == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated)

# Usage
model = VBC_GPT2()
text = model.generate("Once upon a time")
print(text)
```

### Pattern 2: Decision-Making Under Uncertainty

```python
from vbc_prototype import VariableBarrierController
import numpy as np

class RobotDecisionMaker:
    def __init__(self):
        self.vbc = VariableBarrierController(h_star=0.4, top_k=5)

    def choose_action(self, observations, actions):
        """
        observations: dict with sensor data
        actions: list of possible actions [(name, cost, expected_reward), ...]
        """

        # Convert actions to logits based on expected reward
        logits = np.array([reward - cost for name, cost, reward in actions])

        # Add uncertainty
        logits += np.random.randn(len(actions)) * 0.1

        # VBC deliberation
        for tick in range(12):
            result = self.vbc.process_logits(logits)
            if result is not None:
                chosen_action = actions[result.token_id]
                print(f"Chose: {chosen_action[0]} (h={result.hazard:.3f}, χ={self.vbc.context.chi:.3f})")
                return chosen_action

        # No decision → freeze (safety mode)
        return actions[0]  # Default safe action

# Usage
robot = RobotDecisionMaker()
actions = [
    ("move_forward", cost=1.0, reward=2.0),
    ("turn_left", cost=0.5, reward=1.5),
    ("stop", cost=0.0, reward=0.5),
]
choice = robot.choose_action({}, actions)
```

### Pattern 3: Multi-Chain Reasoning

```python
from vbc_prototype import split_stream, join_streams, HourglassContext

class MultiChainReasoner:
    def __init__(self, n_chains=3):
        self.n_chains = n_chains

    def reason(self, problem, strategies):
        """
        problem: str describing the problem
        strategies: list of reasoning approaches
        """

        # Create base context
        base_context = HourglassContext()

        # Split into parallel reasoning chains
        chains = split_stream(base_context, self.n_chains)

        results = []
        for i, (context, strategy) in enumerate(zip(chains, strategies)):
            # Each strategy produces a VBC instance
            vbc = VariableBarrierController(h_star=0.4)

            # Simulate reasoning with strategy
            logits = strategy(problem)  # Returns logits for this strategy

            # Run VBC
            for tick in range(12):
                result = vbc.process_logits(logits)
                if result is not None:
                    results.append((strategy.__name__, result, vbc.context))
                    break

        # Join by weighted hazard
        if results:
            best_strategy, best_result, best_context = max(results,
                                                           key=lambda x: x[1].hazard)
            print(f"Selected strategy: {best_strategy} (h={best_result.hazard:.3f})")
            return best_result

        return None

# Usage
def deductive_strategy(problem):
    # Returns logits favoring logical deduction
    return np.array([0.9, 0.05, 0.05])  # High confidence in option 0

def inductive_strategy(problem):
    # Returns logits favoring pattern matching
    return np.array([0.3, 0.6, 0.1])  # Prefers option 1

reasoner = MultiChainReasoner(n_chains=2)
result = reasoner.reason("Is this valid?", [deductive_strategy, inductive_strategy])
```

---

## Parameter Tuning Guide

### h* (Commit Threshold)

**Default**: 0.5

**Lower h* (0.3-0.4)**:
- Commits faster
- Less deliberation
- Use for: Real-time systems, low-latency applications

**Higher h* (0.6-0.8)**:
- More cautious
- Longer deliberation
- Use for: High-stakes decisions, reasoning tasks

**How to tune**:
```python
# Start with 0.5
vbc = VariableBarrierController(h_star=0.5)

# Monitor commit rate
commits = 0
attempts = 0
for _ in range(100):
    result = vbc.process_logits(logits)
    attempts += 1
    if result is not None:
        commits += 1

commit_rate = commits / attempts

# Adjust
if commit_rate < 0.7:  # Too slow
    h_star -= 0.05
elif commit_rate > 0.95:  # Too fast
    h_star += 0.05
```

### ζ* (Brittleness Threshold)

**Default**: 0.9

**Lower ζ* (0.6-0.8)**:
- More conservative
- Freezes under moderate pressure
- Use for: Safety-critical systems

**Higher ζ* (0.95-1.0)**:
- More tolerant of effort
- Commits even when costly
- Use for: Exploratory systems, research

### κ (Sensitivity)

**Default**: 1.0

Rarely needs tuning. Scales all hazards uniformly.

### top_k (Candidate Count)

**Default**: 10

**Lower top_k (3-5)**:
- Faster computation
- Less exploration
- Use for: Greedy search, exploitation

**Higher top_k (20-50)**:
- More exploration
- Better for uncertain environments
- Use for: Discovery, creativity

---

## Advanced: Custom Hazard Components

You can override VBC's internal methods to customize behavior:

### Custom Epsilon Calculation

```python
from vbc_prototype import VariableBarrierController

class CustomVBC(VariableBarrierController):
    def _compute_epsilon(self, candidate):
        # Your custom logic here
        # Example: epsilon based on attention weights

        # Get attention from model (if available)
        attention = self.get_attention_for_token(candidate.token_id)

        # Epsilon = high attention - dissipation
        epsilon = attention - (1.0 / len(self.context.candidate_futures))

        return max(0.0, epsilon)
```

### Custom Brittleness

```python
class BudgetAwareVBC(VariableBarrierController):
    def __init__(self, *args, compute_budget=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_budget = compute_budget
        self.compute_used = 0

    def _compute_brittleness(self, candidate):
        base_zeta = super()._compute_brittleness(candidate)

        # Add budget pressure
        budget_pressure = self.compute_used / self.compute_budget

        return min(1.0, base_zeta + budget_pressure)

    def process_logits(self, logits):
        self.compute_used += 1  # Track compute
        return super().process_logits(logits)
```

### Custom Alignment

```python
from sentence_transformers import SentenceTransformer

class SemanticVBC(VariableBarrierController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.context_embedding = None

    def set_context(self, context_text):
        self.context_embedding = self.embedder.encode(context_text)

    def _compute_alignment(self, candidate):
        if self.context_embedding is None:
            return super()._compute_alignment(candidate)

        # Compute semantic similarity
        token_embedding = self.embedder.encode(candidate.token_str)
        similarity = cosine_similarity(self.context_embedding, token_embedding)

        return (similarity + 1.0) / 2.0  # Map [-1,1] → [0,1]

# Usage
vbc = SemanticVBC(h_star=0.5)
vbc.set_context("The weather today is")
result = vbc.process_logits(logits)  # Will prefer weather-related tokens
```

---

## Monitoring and Debugging

### Logging Hazard Scores

```python
import logging

logging.basicConfig(level=logging.DEBUG)

class VBCLogger(VariableBarrierController):
    def _commit_phase(self):
        result = super()._commit_phase()

        if result is not None:
            logging.info(f"COMMIT: token={result.token_id}, h={result.hazard:.3f}")
        else:
            hazards = [c.hazard for c in self.context.candidate_futures]
            logging.debug(f"DELIBERATE: max_h={max(hazards):.3f}, threshold={self.h_star:.3f}")

        return result
```

### Visualizing χ Over Time

```python
import matplotlib.pyplot as plt

chi_history = []

for _ in range(100):
    result = vbc.process_logits(logits)
    chi_history.append(vbc.context.chi)

plt.plot(chi_history)
plt.axhline(y=1.0, color='r', linestyle='--', label='Critical (χ=1)')
plt.xlabel('Tick')
plt.ylabel('χ (Criticality)')
plt.title('Phase-Lock Criticality Over Time')
plt.legend()
plt.show()
```

### Debug Mode

```python
vbc = VariableBarrierController(h_star=0.5, top_k=10)

# Enable debug prints
vbc.debug = True

for tick in range(12):
    phase = ['CAPTURE', 'CLEAN', 'BRIDGE', 'COMMIT'][tick % 4]
    print(f"Tick {tick}: {phase}")

    result = vbc.process_logits(logits)

    print(f"  Candidates: {len(vbc.context.candidate_futures)}")
    print(f"  χ: {vbc.context.chi:.3f}")

    if result:
        print(f"  → COMMITTED: h={result.hazard:.3f}")
        break
```

---

## Performance Optimization

### Batch Processing

```python
class BatchVBC:
    def __init__(self, batch_size=8, **vbc_kwargs):
        self.vbcs = [VariableBarrierController(**vbc_kwargs)
                     for _ in range(batch_size)]

    def process_batch(self, logits_batch):
        """
        logits_batch: [batch_size, vocab_size]
        Returns: [batch_size] list of results
        """
        results = []
        for vbc, logits in zip(self.vbcs, logits_batch):
            result = vbc.process_logits(logits)
            results.append(result)
        return results
```

### Caching

```python
from functools import lru_cache

class CachedVBC(VariableBarrierController):
    @lru_cache(maxsize=1000)
    def _compute_epsilon_cached(self, p_value):
        # Epsilon depends only on probability
        return p_value  # Simplified

    def _compute_epsilon(self, candidate):
        return self._compute_epsilon_cached(candidate.p)
```

---

## Testing Your Integration

### Unit Tests

```python
import unittest
from vbc_prototype import VariableBarrierController
import numpy as np

class TestVBCIntegration(unittest.TestCase):
    def test_commit_happens(self):
        vbc = VariableBarrierController(h_star=0.3, top_k=5)
        logits = np.array([10.0] + [0.0]*99)  # Very confident

        result = None
        for _ in range(12):
            result = vbc.process_logits(logits)
            if result is not None:
                break

        self.assertIsNotNone(result, "VBC should commit with high confidence")

    def test_no_premature_commit(self):
        vbc = VariableBarrierController(h_star=0.9, top_k=5)  # High threshold
        logits = np.random.randn(100)  # Uncertain

        result = None
        for _ in range(4):  # Only 1 cycle
            result = vbc.process_logits(logits)

        self.assertIsNone(result, "VBC should not commit prematurely")

    def test_chi_computation(self):
        vbc = VariableBarrierController(h_star=0.5, top_k=10)

        # High-confidence logits
        logits = np.zeros(100)
        logits[0] = 10.0

        for _ in range(3):  # Run to BRIDGE phase
            vbc.process_logits(logits)

        self.assertGreater(vbc.context.chi, 0, "χ should be positive")

if __name__ == '__main__':
    unittest.main()
```

---

## Common Patterns and Best Practices

### DO:
- ✅ Start with default parameters (h*=0.5, ζ*=0.9, top_k=10)
- ✅ Monitor commit rate and χ over time
- ✅ Use multiple chains for complex reasoning
- ✅ Log hazard scores for debugging
- ✅ Test edge cases (very high/low confidence)

### DON'T:
- ❌ Set h* too high (>0.9) unless you want very rare commits
- ❌ Set top_k too low (<3) unless you want pure exploitation
- ❌ Ignore χ warnings (χ > 1 means system is unstable)
- ❌ Skip the CLEAN phase (it filters bad candidates)
- ❌ Use VBC for real-time (<1ms latency) without optimization

---

## Troubleshooting

### Problem: VBC never commits

**Symptoms**: `process_logits()` always returns `None`

**Causes**:
1. h* threshold too high
2. All candidates filtered out in CLEAN phase
3. Brittleness ζ → ζ* (all hazards ≈ 0)

**Solutions**:
```python
# Lower h*
vbc.h_star = 0.3

# Lower CLEAN threshold
# (Modify _clean_phase to keep more candidates)

# Increase ζ*
vbc.zeta_star = 1.0  # No brittleness limit
```

### Problem: VBC commits too quickly

**Symptoms**: Commits in first tick cycle

**Causes**:
1. h* threshold too low
2. Very high-confidence logits

**Solutions**:
```python
# Raise h*
vbc.h_star = 0.7

# Add minimum deliberation time
min_ticks = 4
if tick < min_ticks:
    continue  # Force deliberation
```

### Problem: χ > 1 consistently

**Symptoms**: System is supercritical, unstable

**Causes**:
1. Too many candidates (high flux)
2. Low entropy (low dissipation)

**Solutions**:
```python
# Reduce top_k
vbc.top_k = 5

# Add artificial dissipation
# (Increase damping in epsilon calculation)
```

---

## Production Checklist

Before deploying VBC in production:

- [ ] Tuned h*, ζ*, top_k on validation set
- [ ] Monitored commit rate (target: 70-90%)
- [ ] Checked χ distribution (should be <1 most of the time)
- [ ] Tested edge cases (empty logits, all-zero, etc.)
- [ ] Profiled performance (latency <10ms per token)
- [ ] Added logging and monitoring
- [ ] Implemented fallback for non-commits
- [ ] Tested with real user data
- [ ] Documented parameter choices
- [ ] Created rollback plan

---

## Next Steps

1. **Read**: [MANIFESTO.md](MANIFESTO.md) for complete theory
2. **Experiment**: [vbc_demonstrations.py](vbc_demonstrations.py) for examples
3. **Learn**: [MATHEMATICAL_PROOFS.md](MATHEMATICAL_PROOFS.md) for rigor
4. **Build**: Integrate VBC into your own systems!

---

## Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Share your VBC implementations
- **Twitter**: @YourHandle for updates

---

**Happy Building! 🚀**

*Remember: This is ONE algorithm working across all substrates. The same mathematics that governs quantum measurement also governs your LLM.*
