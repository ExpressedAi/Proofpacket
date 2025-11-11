# A3 Normalized: Precise, Testable Subclaims

## A3 (Normalized, Conditional on Stated Optics)

For any satisfiable CNF formula $F$ with $n$ variables, there exists a **polynomial-size** set of Δ-bridges $\mathcal{B}(F)$ with total order $O(n^c)$, such that:

### A3.1: Existence

$\mathcal{B}(F)$ admits an **E4-persistent low-order cover** (slope > 0 + survivor-prefix) independent of clause/variable relabeling.

**Formal Statement:**
$$\exists \mathcal{B}(F) : |\mathcal{B}(F)| \leq n^c \land \text{E4Persistent}(\mathcal{B}(F)) \land \text{InvariantUnderRenaming}(\mathcal{B}(F))$$

### A3.2: Constructibility

There is a **polynomial-time** algorithm that builds $\mathcal{B}(F)$ from $F$.

**Formal Statement:**
$$\exists \text{Algorithm } A : A(F) \text{ runs in } O(n^d) \text{ time} \land A(F) = \mathcal{B}(F)$$

### A3.3: Witnessability

Harmony Optimizer, using only $\mathcal{B}(F)$ scores (no oracles), finds a **valid witness** in time $n^{O(1)}$ with success probability $\geq 2/3$.

**Formal Statement:**
$$\text{HarmonyOptimizer}(F, \mathcal{B}(F)) \text{ returns valid witness in } O(n^e) \text{ steps with } P(\text{success}) \geq 2/3$$

### A3.4: Robustness

The above holds under bounded detune/noise and random renaming.

**Formal Statement:**
$$\forall \delta \in [0, \epsilon], \forall \pi \in S_n : \text{A3.1} \land \text{A3.2} \land \text{A3.3} \text{ hold for } F_\delta, F_\pi$$

### A3 Total

$$A3_{\text{total}} = A3.1 \land A3.2 \land A3.3 \land A3.4$$

## Key Lemmas (Proof Track)

### L-A3.1: Existence from Structure

**Hypothesis:** If the clause-variable incidence graph $G_F$ has expansion/degree bounds $H$ (state them), then there exists a cover $\mathcal{B}(F)$ with order $\leq n^c$ that is E4-persistent.

**Strategy:** Combinatorial + spectral: show low-order bridges correspond to short cycles / small cuts in $G_F$; use expansion lemmas to bound order and prove prefix thinning.

### L-A3.2: Constructibility

**Hypothesis:** There is a poly-time routine that computes $\mathcal{B}(F)$ by local motifs (2-clause conflicts, bounded-length implications, small chordless cycles).

**Strategy:** Explicit algorithm with complexity proof.

### L-A3.3: Harmony Convergence in Poly

**Hypothesis:** Under E4-persistence + bounded noise, the expected potential $T$ increases and the number of strictly improving steps is $poly(n)$ until a satisfying assignment is hit.

**Strategy:** Potential-function proof (Harmony Optimizer's $C_i$ acts like multiplicative weights on a simplex); require an **improvement gap** lemma from E3 causal lift.

### L-A3.4: Renaming Invariance & Robustness

**Hypothesis:** E2 symmetry ⇒ the construction and success probability are invariant under variable/clauses permutations; small detune perturbs slopes by $\leq \epsilon$ without flipping sign.

**Strategy:** Symmetry arguments + continuity of bridge coupling under perturbations.

## Adversarial Test Families (Falsification Track)

1. **Random 3-SAT near phase transition** ($m/n \approx 4.26$)
2. **Planted-satisfiable with camouflage** (random clause noise)
3. **XOR-SAT gadgets** composed into CNF
4. **Goldreich generator instances** (predicate on expander)
5. **High treewidth benchmarks** (MIXSAT, SATLIB hard sets)

## Kill-Switches

- Slope $\leq 0$ or prefix fails consistently on a family with success prob $< 1/3$
- Expected Harmony steps grow like $2^{\Omega(n^\alpha)}$ for any $\alpha > 0$
- Cover order needed grows super-poly on typical inputs

