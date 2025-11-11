(() => {
    const defaultDataset = {
        critical: [
            { t: 0, magnitude: 1.0 },
            { t: 1, magnitude: 1.0 },
            { t: 2, magnitude: 1.0 },
            { t: 3, magnitude: 1.0 }
        ],
        offCritical: [
            { t: 0, magnitude: 0.61 },
            { t: 1, magnitude: 0.59 },
            { t: 2, magnitude: 0.58 },
            { t: 3, magnitude: 0.6 }
        ]
    };

    const buildPlot = (dataset) => {
        const container = document.getElementById('criticalLinePlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const criticalTrace = {
            x: dataset.critical.map((point) => point.t),
            y: dataset.critical.map((point) => point.magnitude),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'σ = 1/2',
            line: { color: '#1c64f2', width: 3 },
            marker: { color: '#1c64f2', size: 8 }
        };

        const offCriticalTrace = {
            x: dataset.offCritical.map((point) => point.t),
            y: dataset.offCritical.map((point) => point.magnitude),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'σ ≠ 1/2',
            line: { color: '#e02424', width: 2, dash: 'dash' },
            marker: { color: '#e02424', size: 7 }
        };

        const layout = {
            margin: { l: 50, r: 30, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Index (t)' },
            yaxis: { title: 'K₁:₁ fidelity', range: [0.5, 1.05] },
            legend: { orientation: 'h', y: -0.2 }
        };

        Plotly.newPlot(container, [criticalTrace, offCriticalTrace], layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const updateStats = (payload) => {
        if (!payload) {
            return;
        }

        const zeros = payload.riemann?.results?.critical_zeros?.total
            ?? payload.riemann?.results?.total_zeros;
        if (zeros) {
            const el = document.getElementById('stat-critical-zeros');
            if (el) {
                el.textContent = zeros.toLocaleString();
            }
        }

        const rejection = payload.riemann?.results?.e4_rejection_rate;
        if (typeof rejection === 'number') {
            const el = document.getElementById('stat-rejection-rate');
            if (el) {
                el.textContent = `${rejection.toFixed(1)}%`;
            }
        }

        const phaseLock = payload.riemann?.results?.phase_lock_fidelity;
        if (typeof phaseLock === 'number') {
            const el = document.getElementById('stat-phase-lock');
            if (el) {
                el.textContent = `K₁:₁ = ${phaseLock.toFixed(3)}`;
            }
        }

        const validation = payload.riemann?.validation
            ?? payload.validation_report?.problem_results?.riemann;
        if (validation) {
            const trialsEl = document.getElementById('stat-riemann-trials');
            if (trialsEl && typeof validation.n_trials === 'number') {
                trialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            }
            const successEl = document.getElementById('stat-riemann-success');
            if (successEl) {
                const successes = validation.successes ?? validation.passed ?? validation.confirmed;
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
            }
        }
    };

    const extractDataset = (payload) => {
        const critical = payload.riemann?.results?.critical_line_series;
        const offLine = payload.riemann?.results?.off_line_series;

        if (Array.isArray(critical) && Array.isArray(offLine)) {
            return {
                critical: critical.map((point) => ({
                    t: point.t ?? point.index ?? 0,
                    magnitude: point.value ?? point.magnitude ?? 0
                })),
                offCritical: offLine.map((point) => ({
                    t: point.t ?? point.index ?? 0,
                    magnitude: point.value ?? point.magnitude ?? 0
                }))
            };
        }

        return defaultDataset;
    };

    window.addEventListener('DOMContentLoaded', async () => {
        buildPlot(defaultDataset);

        try {
            const response = await fetch('/api/proofs');
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            updateStats(payload);
            buildPlot(extractDataset(payload));
        } catch (error) {
            console.warn('Failed to hydrate Riemann dataset:', error);
        }
    });
})();
