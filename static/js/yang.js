(() => {
    const defaultSpectrum = [
        { channel: '0++', omega: 1.02 },
        { channel: '2++', omega: 1.03 },
        { channel: '1--', omega: 1.05 },
        { channel: '0-+', omega: 1.01 }
    ];

    const updateMetrics = (payload) => {
        if (!payload) {
            return;
        }

        const omegaMin = payload.yang_mills?.results?.omega_min
            ?? payload.yang_mills?.results?.mass_gap_min;
        if (typeof omegaMin === 'number') {
            const el = document.getElementById('stat-omega');
            if (el) {
                el.innerHTML = `ω<sub>min</sub> = ${omegaMin.toFixed(3)}`;
            }
        }

        const channels = payload.yang_mills?.results?.channels_verified;
        if (typeof channels === 'number') {
            const el = document.getElementById('stat-channels');
            if (el) {
                el.textContent = `${channels} / 9`;
            }
        }

        const rgDepth = payload.yang_mills?.results?.rg_depth;
        if (typeof rgDepth === 'number') {
            const el = document.getElementById('stat-rg');
            if (el) {
                el.textContent = `${rgDepth} stages`;
            }
        }

        const validation = payload.yang_mills?.validation
            ?? payload.validation_report?.problem_results?.yang_mills;
        if (validation) {
            const trialsEl = document.getElementById('stat-yang-trials');
            if (trialsEl && typeof validation.n_trials === 'number') {
                trialsEl.textContent = validation.n_trials.toLocaleString('en-US');
            }
            const successEl = document.getElementById('stat-yang-success');
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

    const buildPlot = (spectrum) => {
        const container = document.getElementById('yangMassPlot');
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const trace = {
            x: spectrum.map((item) => item.channel),
            y: spectrum.map((item) => item.omega),
            type: 'bar',
            marker: {
                color: spectrum.map(() => '#1c64f2'),
                line: { width: 1, color: '#123c8c' }
            },
            hovertemplate: 'Channel %{x}<br>ω = %{y:.3f}<extra></extra>'
        };

        const layout = {
            margin: { l: 60, r: 30, t: 30, b: 50 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            yaxis: {
                title: 'ω',
                range: [0.9, 1.1],
                dtick: 0.05
            },
            xaxis: {
                title: 'Channel'
            }
        };

        Plotly.newPlot(container, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    };

    const extractSpectrum = (payload) => {
        const readings = payload.yang_mills?.results?.channel_readings;
        if (Array.isArray(readings)) {
            return readings.map((item) => ({
                channel: item.channel ?? 'channel',
                omega: item.omega ?? item.mass_gap ?? 1
            }));
        }
        return defaultSpectrum;
    };

    window.addEventListener('DOMContentLoaded', async () => {
        buildPlot(defaultSpectrum);

        try {
            const response = await fetch('/api/proofs');
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            updateMetrics(payload);
            buildPlot(extractSpectrum(payload));
        } catch (error) {
            console.warn('Failed to hydrate Yang–Mills dataset:', error);
        }
    });
})();
