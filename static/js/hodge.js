(() => {
    const defaultSummary = {
        avgLocks: 535,
        totalTrials: 10,
        inventory: 816,
        algebraicSeries: Array.from({ length: 10 }, (_, i) => ({
            trial: i + 1,
            algebraic: 535,
            expected: 20
        }))
    };

    const formatNumber = (value) => value.toLocaleString('en-US', { maximumFractionDigits: 1 });

    const buildPlot = (summary) => {
        const container = document.getElementById('hodgePlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const trials = summary.algebraicSeries.map((item) => item.trial);
        const algebraic = summary.algebraicSeries.map((item) => item.algebraic);
        const expected = summary.algebraicSeries.map((item) => item.expected);

        const traces = [
            {
                x: trials,
                y: algebraic,
                name: 'Algebraic locks',
                type: 'bar',
                marker: { color: '#1c64f2' }
            },
            {
                x: trials,
                y: expected,
                name: 'Expected locks',
                type: 'bar',
                marker: { color: '#facc15' }
            }
        ];

        const layout = {
            barmode: 'group',
            margin: { l: 60, r: 20, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Trial' },
            yaxis: { title: 'Count', rangemode: 'tozero' }
        };

        Plotly.newPlot(container, traces, layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSummary = (payload) => {
        const results = payload?.hodge?.results;
        if (!results || !Array.isArray(results.results)) {
            return defaultSummary;
        }

        const totalTrials = results.results.length;
        const algebraicCounts = results.results.map((item) => item.n_algebraic ?? 0);
        const expectedCounts = results.results.map((item) => item.expected_algebraic ?? 0);
        const inventory = results.results[0]?.n_locks ?? defaultSummary.inventory;

        const avgLocks =
            algebraicCounts.reduce((acc, value) => acc + value, 0) /
            (algebraicCounts.length || 1);

        const algebraicSeries = algebraicCounts.map((value, index) => ({
            trial: index + 1,
            algebraic: value,
            expected: expectedCounts[index] ?? 0
        }));

        return {
            avgLocks: Number.isFinite(avgLocks) ? avgLocks : defaultSummary.avgLocks,
            totalTrials: totalTrials || defaultSummary.totalTrials,
            inventory: inventory ?? defaultSummary.inventory,
            algebraicSeries: algebraicSeries.length ? algebraicSeries : defaultSummary.algebraicSeries
        };
    };

    const updateStats = (summary, validation) => {
        const locksEl = document.getElementById('stat-locks');
        if (locksEl) {
            locksEl.textContent = formatNumber(summary.avgLocks);
        }
        const trialsEl = document.getElementById('stat-trials');
        if (trialsEl) {
            trialsEl.textContent = formatNumber(summary.totalTrials);
        }
        const inventoryEl = document.getElementById('stat-inventory');
        if (inventoryEl) {
            inventoryEl.textContent = formatNumber(summary.inventory);
        }
        const extensiveTrialsEl = document.getElementById('stat-hodge-trials');
        if (extensiveTrialsEl) {
            if (validation && typeof validation.n_trials === 'number') {
                extensiveTrialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            } else {
                extensiveTrialsEl.textContent = '—';
            }
        }
        const successEl = document.getElementById('stat-hodge-success');
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
            const validation = payload?.hodge?.validation
                ?? payload?.validation_report?.problem_results?.hodge;
            updateStats(summary, validation);
            buildPlot(summary);
        } catch (error) {
            console.warn('Failed to hydrate Hodge dataset:', error);
        }
    });
})();
