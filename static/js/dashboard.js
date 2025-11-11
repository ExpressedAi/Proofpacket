(() => {
    const setStatValue = (proofKey, statKey, value, options = {}) => {
        if (value == null || Number.isNaN(value)) {
            return;
        }
        const selector = `[data-proof="${proofKey}"] [data-stat="${statKey}"]`;
        const el = document.querySelector(selector);
        if (!el) {
            return;
        }

        if (options.denominator != null) {
            el.dataset.denominator = String(options.denominator);
        }
        if (options.template) {
            el.dataset.template = options.template;
        }

        if (typeof window.animateStatValue === 'function') {
            window.animateStatValue(el, value, options);
            return;
        }

        const format = options.format || el.dataset.animate;
        const decimals = Number(options.decimals ?? el.dataset.decimals ?? 0);

        if (format === 'percent') {
            el.textContent = `${(value * 100).toFixed(decimals)}%`;
            return;
        }

        if (format === 'ratio') {
            const denominator = options.denominator ?? Number(el.dataset.denominator ?? 0);
            const numeratorText = Number(value).toLocaleString('en-US', {
                maximumFractionDigits: decimals,
                minimumFractionDigits: decimals
            });
            if (denominator) {
                const denominatorText = Number(denominator).toLocaleString('en-US', {
                    maximumFractionDigits: decimals,
                    minimumFractionDigits: decimals
                });
                el.textContent = `${numeratorText} / ${denominatorText}`;
            } else {
                el.textContent = numeratorText;
            }
            return;
        }

        if (format === 'template') {
            const template = options.template || el.dataset.template || '{value}';
            el.innerHTML = template.replace('{value}', Number(value).toFixed(decimals));
            return;
        }

        el.textContent = Number(value).toLocaleString('en-US', {
            maximumFractionDigits: decimals,
        });
    };

    const updateStats = (data) => {
        if (!data) {
            return;
        }

        const riemannZeros = data.riemann?.results?.critical_zeros?.total
            ?? data.riemann?.results?.total_zeros
            ?? null;
        if (typeof riemannZeros === 'number') {
            setStatValue('riemann', 'zeros', riemannZeros);
        }

        const navierSummary = data.navier_stokes?.results?.summary
            ?? data.navier_stokes?.summary
            ?? data.navier_stokes?.extended_results?.summary
            ?? null;
        if (navierSummary) {
            const smooth = navierSummary.smooth_count ?? navierSummary.runs_completed ?? navierSummary.total_tests;
            const total = navierSummary.total_tests ?? navierSummary.runs_total ?? smooth;
            if (typeof smooth === 'number') {
                setStatValue('navier', 'simulations', smooth, { denominator: total });
            }
        }

        const massGap = data.yang_mills?.results?.mass_gap_min
            ?? data.yang_mills?.results?.omega_min
            ?? null;
        if (typeof massGap === 'number') {
            setStatValue('yang', 'mass-gap', massGap);
        }

        const pvsnpRate = data.p_vs_np?.results?.summary?.poly_cover_rate;
        if (typeof pvsnpRate === 'number') {
            setStatValue('p-vs-np', 'poly-cover', pvsnpRate);
        }

        const bsdRank = data.bsd?.results?.summary?.avg_rank_estimate;
        if (typeof bsdRank === 'number') {
            setStatValue('bsd', 'rank', bsdRank);
        }

        const hodgeLocks = data.hodge?.results?.summary?.avg_algebraic_cycles;
        if (typeof hodgeLocks === 'number') {
            setStatValue('hodge', 'locks', hodgeLocks);
        }

        const poincareTrivial = data.poincare?.results?.summary?.confirmed_rate;
        if (typeof poincareTrivial === 'number') {
            setStatValue('poincare', 'holonomy', poincareTrivial);
        }

        const applyValidationText = (cardKey, dataKey) => {
            const validation = data[dataKey]?.validation;
            const el = document.querySelector(`[data-proof="${cardKey}"] [data-stat="validation"]`);
            if (!el) {
                return;
            }
            if (!validation || typeof validation !== 'object') {
                el.textContent = 'Extensive validation: pending';
                return;
            }
            const successes = validation.successes ?? validation.confirmed ?? validation.passed;
            const trials = validation.n_trials ?? validation.total_trials ?? validation.trials;
            const rate = validation.success_rate ?? validation.rate;
            if (typeof successes === 'number' && typeof trials === 'number') {
                const successText = successes.toLocaleString('en-US');
                const trialText = trials.toLocaleString('en-US');
                const rateText = typeof rate === 'number' ? ` (${(rate * 100).toFixed(1)}%)` : '';
                el.textContent = `Extensive validation: ${successText} / ${trialText} trials passed${rateText}`;
            } else {
                el.textContent = 'Extensive validation: data unavailable';
            }
        };

        applyValidationText('riemann', 'riemann');
        applyValidationText('navier', 'navier_stokes');
        applyValidationText('yang', 'yang_mills');
        applyValidationText('p-vs-np', 'p_vs_np');
        applyValidationText('bsd', 'bsd');
        applyValidationText('hodge', 'hodge');
        applyValidationText('poincare', 'poincare');

        const lastSync = data.manifest?.generated_at
            ?? data.manifest?.updated_at
            ?? new Date().toISOString();
        const syncEl = document.getElementById('last-sync');
        if (syncEl) {
            try {
                const parsed = new Date(lastSync);
                syncEl.textContent = parsed.toLocaleString();
            } catch {
                syncEl.textContent = lastSync;
            }
        }
    };

    const initIntersectionAnimations = () => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.15
        });

        document.querySelectorAll('.panel, .card, .glass').forEach((el) => {
            observer.observe(el);
        });
    };

    window.addEventListener('DOMContentLoaded', async () => {
        initIntersectionAnimations();

        try {
            const response = await fetch('/api/proofs');
            if (response.ok) {
                const payload = await response.json();
                updateStats(payload);
            } else {
                console.warn('Failed to load proof data:', response.status);
            }
        } catch (error) {
            console.warn('Error fetching proof data:', error);
        }
    });
})();
