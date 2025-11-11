-- Formal E4 Wiring: Coarse-Grain Persistence
-- Lemma NS-E4: χ bound persists under ×2 aggregation

import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic
import ns_proof  -- Import constants and bound lemmas

namespace NavierStokes

-- Re-export constants from ns_proof
open NavierStokes (C_B C_T C_R C_com C1 C2 C3 c_nu delta_structural chi D chiM Π_nloc_gt D_aggregated)

-- Aggregated shell index (block size 2)
def aggregated_shell (j : ℤ) : ℤ := j / 2

-- Aggregated flux (sum over shells 2j and 2j+1)
def Π_aggregated (u : SmoothSolution) (j_bar : ℤ) (t : ℝ) : ℝ :=
  Π u (2 * j_bar) t + Π u (2 * j_bar + 1) t

def Π_nloc_aggregated (u : SmoothSolution) (j_bar : ℤ) (t : ℝ) : ℝ :=
  Π_nloc u (2 * j_bar) t + Π_nloc u (2 * j_bar + 1) t

def Π_loc_aggregated (u : SmoothSolution) (j_bar : ℤ) (t : ℝ) : ℝ :=
  Π_loc u (2 * j_bar) t + Π_loc u (2 * j_bar + 1) t + cross_shell_local u (2 * j_bar) t

-- Cross-shell local terms (between adjacent shells in same block)
def cross_shell_local (u : SmoothSolution) (j : ℤ) (t : ℝ) : ℝ :=
  -- Local interactions between shells j and j+1 (|j - (j+1)| = 1 ≤ 1, so local)
  -- This is part of the aggregated local flux
  0  -- TODO: Define explicitly from triad interactions

-- Aggregated dissipation
def D_aggregated (u : SmoothSolution) (j_bar : ℤ) (t : ℝ) : ℝ :=
  D u (2 * j_bar) t + D u (2 * j_bar + 1) t

-- Note: chiM_aggregated not needed for dissipation-sum argument, but kept for reference

-- Coarse-grain persistence lemma (finite-band version)
-- Uses the correct dissipation-sum argument from the LaTeX proof
theorem coarse_grain_persistence
  (u : SmoothSolution) (t : ℝ) (M : ℤ) (η : ℝ) (hM : M ≥ 1) (hη : 0 < η ∧ η < 1)
  (h_locality : ∀ j, Real.max (Π_nloc_gt u M j t) 0 ≤ η * D u j t)
  (h_dissipation_nonneg : ∀ j, 0 ≤ D u j t) :
  ∀ J, Real.max (Π_nloc_gt_aggregated u M J t) 0 ≤ η * D_aggregated u J t := by
  intro J
  -- Define aggregated quantities
  have h_nloc_def : Π_nloc_gt_aggregated u M J t = Π_nloc_gt u M (2 * J) t + Π_nloc_gt u M (2 * J + 1) t := by
    rfl
  have h_D_def : D_aggregated u J t = D u (2 * J) t + D u (2 * J + 1) t := by
    rfl
  
  -- Use subadditivity of max: max(a+b, 0) ≤ max(a,0) + max(b,0)
  have h_num_subadd : 
    Real.max (Π_nloc_gt u M (2 * J) t + Π_nloc_gt u M (2 * J + 1) t) 0 ≤
    Real.max (Π_nloc_gt u M (2 * J) t) 0 + Real.max (Π_nloc_gt u M (2 * J + 1) t) 0 := by
    -- Standard fact: max(x+y, 0) ≤ max(x,0) + max(y,0)
    exact max_add_nonpos_le (Π_nloc_gt u M (2 * J) t) (Π_nloc_gt u M (2 * J + 1) t)
  
  -- Apply locality hypothesis to each shell
  have h_bound1 : Real.max (Π_nloc_gt u M (2 * J) t) 0 ≤ η * D u (2 * J) t := by
    exact h_locality (2 * J)
  have h_bound2 : Real.max (Π_nloc_gt u M (2 * J + 1) t) 0 ≤ η * D u (2 * J + 1) t := by
    exact h_locality (2 * J + 1)
  
  -- Sum the bounds
  have h_sum_of_bounds :
    Real.max (Π_nloc_gt u M (2 * J) t) 0 + Real.max (Π_nloc_gt u M (2 * J + 1) t) 0 ≤
    η * D u (2 * J) t + η * D u (2 * J + 1) t := by
    exact add_le_add h_bound1 h_bound2
  
  -- Combine: max(nloc_agg, 0) ≤ sum of max(nloc_i, 0) ≤ sum of η*D_i = η * D_agg
  calc
    Real.max (Π_nloc_gt_aggregated u M J t) 0
      = Real.max (Π_nloc_gt u M (2 * J) t + Π_nloc_gt u M (2 * J + 1) t) 0 := by rw [h_nloc_def]
    _ ≤ Real.max (Π_nloc_gt u M (2 * J) t) 0 + Real.max (Π_nloc_gt u M (2 * J + 1) t) 0 := h_num_subadd
    _ ≤ η * D u (2 * J) t + η * D u (2 * J + 1) t := h_sum_of_bounds
    _ = η * (D u (2 * J) t + D u (2 * J + 1) t) := by rw [mul_add]
    _ = η * D_aggregated u J t := by rw [h_D_def]

end NavierStokes

