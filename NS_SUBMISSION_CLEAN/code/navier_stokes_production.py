#!/usr/bin/env python3
"""
Production-Scale Navier-Stokes Test
Multiple viscosities, initial conditions, and resolutions
"""

import numpy as np
import json
from datetime import datetime
from navier_stokes_simple import ShellModel, TriadLockDetector, NavierStokesAuditSuite
import warnings
warnings.filterwarnings('ignore')


class ProductionNavierStokesTest:
    """
    Production-scale test suite
    Multiple parameter combinations
    """
    
    def __init__(self):
        # Pre-registered test parameters
        self.nu_values = [0.001, 0.01, 0.1]  # Viscosities
        self.n_shells_values = [16, 24, 32]  # Resolutions
        self.n_steps = 5000  # Long runs
        self.window_size = 100
        self.detector = TriadLockDetector()
        self.auditor = NavierStokesAuditSuite()
    
    def run_test_suite(self):
        """Run full production test"""
        print("="*80)
        print("PRODUCTION-SCALE NAVIER-STOKES TEST")
        print("="*80)
        print(f"\nStarted: {datetime.now()}")
        
        print(f"\nTest parameters:")
        print(f"  Viscosities: {self.nu_values}")
        print(f"  Shell counts: {self.n_shells_values}")
        print(f"  Steps per run: {self.n_steps}")
        
        all_results = []
        
        for nu in self.nu_values:
            for n_shells in self.n_shells_values:
                print(f"\n{'-'*80}")
                print(f"Testing: ν={nu}, shells={n_shells}")
                print(f"{'-'*80}")
                
                # Initialize model
                model = ShellModel(n_shells=n_shells, nu=nu, dt=0.001)
                detector = TriadLockDetector(nu)
                
                # Run simulation
                all_phasors = []
                all_locks = []
                
                for step in range(self.n_steps):
                    model.step()
                    
                    # Extract phasors periodically
                    if step % self.window_size == 0:
                        phasors = model.get_shell_phasors()
                        locks = detector.detect_triad_locks(phasors)
                        
                        all_phasors.append(phasors)
                        all_locks.append(locks)
                    
                    # Progress indicator
                    if step % 1000 == 0:
                        print(f"  Step {step}/{self.n_steps}")
                
                # Average locks
                avg_locks = self._average_locks(all_locks)
                
                # Run audits
                avg_phasors = all_phasors[-1] if all_phasors else []
                verdict, audits = self.auditor.run_all_audits(avg_locks, avg_phasors)
                
                # Check for supercritical triads
                supercritical = [l for l in avg_locks if l['chi'] > 1.0]
                
                result = {
                    'nu': nu,
                    'n_shells': n_shells,
                    'verdict': verdict,
                    'n_locks': len(avg_locks),
                    'n_eligible': sum(1 for l in avg_locks if l['eligible']),
                    'n_supercritical': len(supercritical),
                    'max_K': float(max([l['K'] for l in avg_locks])) if avg_locks else 0.0,
                    'mean_chi': float(np.mean([l['chi'] for l in avg_locks])) if avg_locks else 0.0,
                    'audits': {k: {'passed': bool(v[0]), 'message': str(v[1])} for k, v in audits.items()},
                    'supercritical_indices': [l['n'] for l in supercritical]
                }
                
                all_results.append(result)
                
                print(f"\n  Verdict: {verdict}")
                print(f"  Eligible locks: {result['n_eligible']}/{result['n_locks']}")
                print(f"  Supercritical triads: {len(supercritical)}")
                if supercritical:
                    print(f"  Shell indices: {result['supercritical_indices']}")
        
        # Summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        
        smooth_count = sum(1 for r in all_results if r['verdict'] == 'SMOOTH')
        total_tests = len(all_results)
        
        print(f"\nTotal tests: {total_tests}")
        print(f"SMOOTH verdicts: {smooth_count} ({100*smooth_count/total_tests:.1f}%)")
        print(f"\nParameter combinations with supercritical triads:")
        
        for r in all_results:
            if r['n_supercritical'] > 0:
                print(f"  ✗ ν={r['nu']}, shells={r['n_shells']}: {r['n_supercritical']} supercritical")
            else:
                print(f"  ✓ ν={r['nu']}, shells={r['n_shells']}: SMOOTH")
        
        # Save results
        report = {
            'parameters': {
                'nu_values': self.nu_values,
                'n_shells_values': self.n_shells_values,
                'n_steps': self.n_steps
            },
            'results': all_results,
            'summary': {
                'total_tests': total_tests,
                'smooth_count': smooth_count,
                'smooth_rate': smooth_count / total_tests if total_tests > 0 else 0
            }
        }
        
        with open("navier_stokes_production_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Results saved to: navier_stokes_production_results.json")
        print(f"✓ Completed: {datetime.now()}")
        
        return report
    
    @staticmethod
    def _average_locks(all_locks):
        """Average lock strengths across windows"""
        if not all_locks:
            return []
        
        triad_dict = {}
        for window_locks in all_locks:
            for lock in window_locks:
                n = lock['n']
                if n not in triad_dict:
                    triad_dict[n] = []
                triad_dict[n].append(lock)
        
        avg_locks = []
        for n in sorted(triad_dict.keys()):
            locks_n = triad_dict[n]
            avg_lock = {
                'n': n,
                'triad': locks_n[0]['triad'],
                'K': np.mean([l['K'] for l in locks_n]),
                'epsilon_cap': np.mean([l['epsilon_cap'] for l in locks_n]),
                'chi': np.mean([l['chi'] for l in locks_n]),
                'zeta': np.mean([l['zeta'] for l in locks_n]),
                'eligible': any(l['eligible'] for l in locks_n)
            }
            avg_locks.append(avg_lock)
        
        return avg_locks


if __name__ == "__main__":
    test = ProductionNavierStokesTest()
    test.run_test_suite()

