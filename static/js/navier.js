(() => {
    const defaultEnergy = {
        time: Array.from({ length: 10 }, (_, i) => i * 5000),
        energy: [1.0, 0.82, 0.67, 0.54, 0.43, 0.34, 0.27, 0.22, 0.19, 0.16]
    };

    const updateMetrics = (payload) => {
        if (!payload) {
            return;
        }

        const totalConfigs = payload.navier_stokes?.results?.configurations
            ?? payload.navier_stokes?.results?.runs_completed;
        if (totalConfigs) {
            const el = document.getElementById('stat-configurations');
            if (el) {
                el.textContent = `${totalConfigs} / 9`;
            }
        }

        const triadMax = payload.navier_stokes?.results?.chi_max;
        if (typeof triadMax === 'number') {
            const el = document.getElementById('stat-triad');
            if (el) {
                el.textContent = triadMax.toExponential(2);
            }
        }

        const timeSteps = payload.navier_stokes?.results?.time_steps;
        if (typeof timeSteps === 'number') {
            const el = document.getElementById('stat-steps');
            if (el) {
                el.textContent = timeSteps.toLocaleString();
            }
        }

        const validation = payload.navier_stokes?.validation
            ?? payload.validation_report?.problem_results?.navier_stokes;
        if (validation) {
            const trialsEl = document.getElementById('stat-navier-trials');
            if (trialsEl && typeof validation.n_trials === 'number') {
                trialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            }
            const successEl = document.getElementById('stat-navier-success');
            if (successEl) {
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
            }
        }
    };

    const buildPlot = (series) => {
        const container = document.getElementById('navierEnergyPlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const trace = {
            x: series.time,
            y: series.energy,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#0e9f6e', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(14, 159, 110, 0.2)',
            name: 'Energy envelope'
        };

        const layout = {
            margin: { l: 50, r: 30, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: { title: 'Time step' },
            yaxis: { title: 'Normalised energy', range: [0, 1.05] },
            showlegend: false
        };

        Plotly.newPlot(container, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSeries = (payload) => {
        const series = payload.navier_stokes?.results?.energy_series;
        if (Array.isArray(series)) {
            return {
                time: series.map((point) => point.t ?? point.time ?? 0),
                energy: series.map((point) => point.value ?? point.energy ?? 0)
            };
        }
        return defaultEnergy;
    };

    window.addEventListener('DOMContentLoaded', async () => {
        buildPlot(defaultEnergy);

        try {
            const response = await fetch('/api/proofs');
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            updateMetrics(payload);
            buildPlot(extractSeries(payload));
        } catch (error) {
            console.warn('Failed to hydrate Navier–Stokes dataset:', error);
        }
    });
})();
