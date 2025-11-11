(() => {
    const defaultSummary = {
        polyCoverRate: 0.5333,
        totalTests: 15,
        bridgesPerInstance: 4400,
        verdicts: {
            POLY_COVER: 8,
            DELTA_BARRIER: 7
        }
    };

    const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
    const formatNumber = (value) => value.toLocaleString('en-US');

    const buildPlot = (summary) => {
        const container = document.getElementById('pVsNpPlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const verdictLabels = Object.keys(summary.verdicts);
        const counts = verdictLabels.map((label) => summary.verdicts[label] ?? 0);

        const data = [
            {
                x: verdictLabels,
                y: counts,
                type: 'bar',
                marker: {
                    color: ['#1c64f2', '#f05252', '#14b8a6', '#facc15']
                }
            }
        ];

        const layout = {
            margin: { l: 50, r: 20, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Verdict' },
            yaxis: { title: 'Count', rangemode: 'tozero' }
        };

        Plotly.newPlot(container, data, layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSummary = (payload) => {
        const results = payload?.p_vs_np?.results;
        if (!results || typeof results !== 'object') {
            return defaultSummary;
        }

        const summary = results.summary ?? {};
        const totalTests = summary.total_tests ?? defaultSummary.totalTests;
        const polyRate = summary.poly_cover_rate ?? defaultSummary.polyCoverRate;
        const bridges = results.results?.[0]?.n_bridges ?? defaultSummary.bridgesPerInstance;

        const verdictTallies = results.results?.reduce((acc, item) => {
            const verdict = item.verdict ?? 'UNKNOWN';
            acc[verdict] = (acc[verdict] ?? 0) + 1;
            return acc;
        }, {}) ?? defaultSummary.verdicts;

        return {
            polyCoverRate: polyRate,
            totalTests,
            bridgesPerInstance: bridges,
            verdicts: verdictTallies
        };
    };

    const updateStats = (summary, validation) => {
        const coverRateEl = document.getElementById('stat-cover-rate');
        if (coverRateEl) {
            coverRateEl.textContent = formatPercent(summary.polyCoverRate);
        }
        const testEl = document.getElementById('stat-total-tests');
        if (testEl) {
            testEl.textContent = formatNumber(summary.totalTests);
        }
        const bridgeEl = document.getElementById('stat-bridges');
        if (bridgeEl) {
            bridgeEl.textContent = formatNumber(summary.bridgesPerInstance);
        }
        const trialsEl = document.getElementById('stat-pvsnp-trials');
        if (trialsEl) {
            if (validation && typeof validation.n_trials === 'number') {
                trialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            } else {
                trialsEl.textContent = '—';
            }
        }
        const successEl = document.getElementById('stat-pvsnp-success');
        if (successEl) {
            if (validation) {
                const successes = validation.successes ?? validation.passed;
                const trials = validation.n_trials;
                const rate = validation.success_rate;
                if (typeof successes === 'number' && typeof trials === 'number' && trials > 0) {
                    const rateText = typeof rate === 'number'
                        ? ` (${(rate * 100).toFixed(1)}%)`
                        : '';
                    successEl.textContent = `${successes.toLocaleString('en-US')} / ${trials.toLocaleString('en-US')} trials passed${rateText}`;
                } else {
                    successEl.textContent = 'Validation data unavailable';
                }
            } else {
                successEl.textContent = 'Awaiting refresh';
            }
        }
    };

    window.addEventListener('DOMContentLoaded', async () => {
        updateStats(defaultSummary, null);
        buildPlot(defaultSummary);

        try {
            const response = await fetch('/api/proofs');
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            const summary = extractSummary(payload);
            const validation = payload?.p_vs_np?.validation
                ?? payload?.validation_report?.problem_results?.p_vs_np;
            updateStats(summary, validation);
            buildPlot(summary);
        } catch (error) {
            console.warn('Failed to hydrate P vs NP dataset:', error);
        }
    });
})();
