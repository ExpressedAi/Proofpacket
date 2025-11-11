-- Helper lemmas for MWU proofs (can be imported or proved)

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

namespace PvsNP

-- Helper: exp bound via convexity for bounded x
/-- Convexity bound for exp on bounded interval. -/
lemma exp_bound_mix (η B x : ℝ) (hB_pos : 0 < B) (hxB : -B ≤ x ∧ x ≤ B) :
  Real.exp (η * x)
    ≤ ((B + x) / (2*B)) * Real.exp (η*B)
    + ((B - x) / (2*B)) * Real.exp (-η*B) := by
  -- convexity of exp at points −B and +B; write x as convex combo
  -- construct θ = (B+x)/(2B) ∈ [0,1], so x = θ*(+B) + (1-θ)*(-B)
  -- then exp(ηx) ≤ θ exp(ηB) + (1-θ) exp(-ηB) by convexity of exp
  -- This is standard convexity of exp; can be proved or imported
  sorry  -- TODO: Prove convexity bound or import from mathlib

-- Helper: cosh ≤ exp(t²/2)
/-- Standard bound: cosh t ≤ exp(t²/2). -/
lemma cosh_le_exp_sq_div_two (t : ℝ) : Real.cosh t ≤ Real.exp (t*t/2) := by
  -- monotone comparison via derivative or power series upper bound
  -- This is a classical inequality
  sorry  -- TODO: Prove or import from mathlib

end PvsNP

