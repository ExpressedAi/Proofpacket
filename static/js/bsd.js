(() => {
    const defaultSummary = {
        avgRank: 2.0,
        avgPersistent: 300,
        totalTrials: 10,
        series: Array.from({ length: 10 }, (_, i) => ({
            trial: i + 1,
            persistent: 300
        }))
    };

    const formatNumber = (value, digits = 0) =>
        Number.parseFloat(value).toLocaleString('en-US', {
            maximumFractionDigits: digits,
            minimumFractionDigits: digits
        });

    const buildPlot = (summary) => {
        const container = document.getElementById('bsdPlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const trace = {
            x: summary.series.map((item) => item.trial),
            y: summary.series.map((item) => item.persistent),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Persistent generators',
            line: { color: '#1c64f2', width: 3 },
            marker: { size: 8 }
        };

        const layout = {
            margin: { l: 60, r: 20, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Trial' },
            yaxis: { title: 'Persistent generators', rangemode: 'tozero' }
        };

        Plotly.newPlot(container, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSummary = (payload) => {
        const results = payload?.bsd?.results;
        if (!results || !Array.isArray(results.results)) {
            return defaultSummary;
        }

        const trials = results.results.length;
        const persistentValues = results.results.map((item) => item.n_persistent ?? 0);

        const avgPersistent =
            persistentValues.reduce((acc, value) => acc + value, 0) / (persistentValues.length || 1);

        const avgRank =
            results.results.reduce((acc, item) => acc + (item.rank_estimate ?? 0), 0) /
            (trials || 1);

        const series = persistentValues.map((value, index) => ({
            trial: index + 1,
            persistent: value
        }));

        return {
            avgRank: Number.isFinite(avgRank) ? avgRank : defaultSummary.avgRank,
            avgPersistent: Number.isFinite(avgPersistent) ? avgPersistent : defaultSummary.avgPersistent,
            totalTrials: trials || defaultSummary.totalTrials,
            series: series.length ? series : defaultSummary.series
        };
    };

    const updateStats = (summary, validation) => {
        const rankEl = document.getElementById('stat-rank');
        if (rankEl) {
            rankEl.textContent = formatNumber(summary.avgRank, 1);
        }
        const persistentEl = document.getElementById('stat-persistent');
        if (persistentEl) {
            persistentEl.textContent = formatNumber(summary.avgPersistent, 0);
        }
        const trialsEl = document.getElementById('stat-trials');
        if (trialsEl) {
            trialsEl.textContent = formatNumber(summary.totalTrials, 0);
        }
        const extensiveTrialsEl = document.getElementById('stat-bsd-trials');
        if (extensiveTrialsEl) {
            if (validation && typeof validation.n_trials === 'number') {
                extensiveTrialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            } else {
                extensiveTrialsEl.textContent = '—';
            }
        }
        const successEl = document.getElementById('stat-bsd-success');
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
            const validation = payload?.bsd?.validation
                ?? payload?.validation_report?.problem_results?.bsd;
            updateStats(summary, validation);
            buildPlot(summary);
        } catch (error) {
            console.warn('Failed to hydrate BSD dataset:', error);
        }
    });
})();
