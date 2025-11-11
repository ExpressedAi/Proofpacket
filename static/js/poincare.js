(() => {
    const defaultSummary = {
        trivialRate: 0,
        locksPerTrial: 2090,
        cycles: 10,
        zeroSeries: Array.from({ length: 10 }, (_, i) => ({
            trial: i + 1,
            zeroRatio: 0.5
        }))
    };

    const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
    const formatNumber = (value) => value.toLocaleString('en-US');

    const buildPlot = (summary) => {
        const container = document.getElementById('poincarePlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const trace = {
            x: summary.zeroSeries.map((item) => item.trial),
            y: summary.zeroSeries.map((item) => item.zeroRatio * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Zero holonomy ratio (%)',
            line: { color: '#1c64f2', width: 3 },
            marker: { size: 8 }
        };

        const layout = {
            margin: { l: 60, r: 20, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Trial' },
            yaxis: { title: 'Zero holonomy (%)', rangemode: 'tozero', range: [0, 100] }
        };

        Plotly.newPlot(container, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSummary = (payload) => {
        const results = payload?.poincare?.results;
        if (!results || !Array.isArray(results.results)) {
            return defaultSummary;
        }

        const trials = results.results.length;
        const locks = results.results[0]?.n_locks ?? defaultSummary.locksPerTrial;
        const cycles = results.results[0]?.n_cycles ?? defaultSummary.cycles;

        const trivialCount = results.results.filter((item) => item.all_m_zero === true).length;

        const zeroSeries = results.results.map((item, index) => {
            const holonomies = Array.isArray(item.holonomies) ? item.holonomies : [];
            const zeroCount = holonomies.filter((value) => value === 0).length;
            const ratio = holonomies.length ? zeroCount / holonomies.length : 0;
            return {
                trial: index + 1,
                zeroRatio: ratio
            };
        });

        return {
            trivialRate: trials ? trivialCount / trials : defaultSummary.trivialRate,
            locksPerTrial: locks,
            cycles,
            zeroSeries: zeroSeries.length ? zeroSeries : defaultSummary.zeroSeries
        };
    };

    const updateStats = (summary, validation) => {
        const holonomyEl = document.getElementById('stat-holonomy');
        if (holonomyEl) {
            holonomyEl.textContent = formatPercent(summary.trivialRate);
        }
        const locksEl = document.getElementById('stat-locks');
        if (locksEl) {
            locksEl.textContent = formatNumber(summary.locksPerTrial);
        }
        const cyclesEl = document.getElementById('stat-cycles');
        if (cyclesEl) {
            cyclesEl.textContent = formatNumber(summary.cycles);
        }
        const trialsEl = document.getElementById('stat-poincare-trials');
        if (trialsEl) {
            if (validation && typeof validation.n_trials === 'number') {
                trialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            } else {
                trialsEl.textContent = '—';
            }
        }
        const successEl = document.getElementById('stat-poincare-success');
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
            const validation = payload?.poincare?.validation
                ?? payload?.validation_report?.problem_results?.poincare;
            updateStats(summary, validation);
            buildPlot(summary);
        } catch (error) {
            console.warn('Failed to hydrate Poincaré dataset:', error);
        }
    });
})();
