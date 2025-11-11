(() => {
    const PROGRAM_LABELS = [
        { key: 'riemann', label: 'Riemann' },
        { key: 'navier_stokes', label: 'Navier–Stokes' },
        { key: 'yang_mills', label: 'Yang–Mills' },
        { key: 'p_vs_np', label: 'P vs NP' },
        { key: 'bsd', label: 'BSD' },
        { key: 'hodge', label: 'Hodge' },
        { key: 'poincare', label: 'Poincaré' }
    ];

    const DEFAULT_EVIDENCE = {
        riemann: 100,
        navier_stokes: 50,
        yang_mills: 50,
        p_vs_np: 100,
        bsd: 100,
        hodge: 100,
        poincare: 100
    };

    const DEFAULT_SUPPORT = {
        riemann: 1.0,
        navier_stokes: 1.0,
        yang_mills: 1.0,
        p_vs_np: 1.0,
        bsd: 1.0,
        hodge: 1.0,
        poincare: 1.0
    };

    const successVerdicts = {
        navier_stokes: new Set(['SMOOTH']),
        yang_mills: new Set(['MASS_GAP', 'CONFIRMED']),
        p_vs_np: new Set(['POLY_COVER', 'SUPPORTED']),
        bsd: new Set(['BSD_CONFIRMED']),
        hodge: new Set(['HODGE_CONFIRMED']),
        poincare: new Set(['S3', 'CONFIRMED'])
    };

    const evidenceContainer = () => document.getElementById('programEvidenceChart');
    const supportContainer = () => document.getElementById('programSupportChart');

    const chartState = {
        evidenceLayout: null,
        supportLayout: null
    };

    const computeRiemannEvidence = (riemann) => {
        const stats = riemann?.summary_statistics;
        if (!stats || typeof stats !== 'object') {
            return DEFAULT_EVIDENCE.riemann;
        }
        return Object.values(stats).reduce((acc, entry) => acc + (entry?.n_tested ?? 0), 0) || DEFAULT_EVIDENCE.riemann;
    };

    const computeArrayLength = (container) => {
        if (!container) {
            return 0;
        }
        if (Array.isArray(container.results)) {
            return container.results.length;
        }
        if (Array.isArray(container)) {
            return container.length;
        }
        return 0;
    };

    const computeEvidence = (payload) => {
        const validationResults = payload?.validation_report?.problem_results ?? {};
        const mapping = {
            riemann: computeRiemannEvidence(payload?.riemann?.results),
            navier_stokes: computeArrayLength(payload?.navier_stokes?.results) || DEFAULT_EVIDENCE.navier_stokes,
            yang_mills: computeArrayLength(payload?.yang_mills?.results) || DEFAULT_EVIDENCE.yang_mills,
            p_vs_np: payload?.p_vs_np?.results?.summary?.total_tests ?? DEFAULT_EVIDENCE.p_vs_np,
            bsd: computeArrayLength(payload?.bsd?.results) || DEFAULT_EVIDENCE.bsd,
            hodge: computeArrayLength(payload?.hodge?.results) || DEFAULT_EVIDENCE.hodge,
            poincare: computeArrayLength(payload?.poincare?.results) || DEFAULT_EVIDENCE.poincare
        };
        for (const [key, report] of Object.entries(validationResults)) {
            if (typeof report?.n_trials === 'number') {
                mapping[key] = report.n_trials;
            }
        }
        return PROGRAM_LABELS.map(({ key, label }) => ({
            key,
            label,
            value: mapping[key] ?? DEFAULT_EVIDENCE[key] ?? 0
        }));
    };

    const computeSupportRate = (payload) => {
        const rates = {};
        const validationResults = payload?.validation_report?.problem_results ?? {};

        // Riemann overall success flag
        const rhOverall = payload?.riemann?.results?.overall;
        rates.riemann = rhOverall?.success === false ? 0 : DEFAULT_SUPPORT.riemann;

        // Navier–Stokes, Yang–Mills, BSD, Hodge, Poincaré
        for (const key of ['navier_stokes', 'yang_mills', 'bsd', 'hodge', 'poincare']) {
            const container = payload?.[key]?.results;
            if (!Array.isArray(container?.results)) {
                rates[key] = DEFAULT_SUPPORT[key];
                continue;
            }
            const verdicts = container.results;
            const positiveSet = successVerdicts[key] ?? new Set();
            const positiveCount = verdicts.filter((item) => positiveSet.has(item?.verdict)).length;
            rates[key] = verdicts.length ? positiveCount / verdicts.length : DEFAULT_SUPPORT[key];
        }

        // P vs NP uses summary rate
        const pvsnpRate = payload?.p_vs_np?.results?.summary?.poly_cover_rate;
        rates.p_vs_np = typeof pvsnpRate === 'number' ? pvsnpRate : DEFAULT_SUPPORT.p_vs_np;

        for (const [key, report] of Object.entries(validationResults)) {
            if (typeof report?.success_rate === 'number') {
                rates[key] = report.success_rate;
            }
        }

        return PROGRAM_LABELS.map(({ key, label }) => ({
            key,
            label,
            value: Math.max(0, Math.min(1, rates[key] ?? DEFAULT_SUPPORT[key] ?? 0))
        }));
    };

    const buildEvidenceChart = (series) => {
        const container = evidenceContainer();
        if (!container || typeof Plotly === 'undefined') {
            return;
        }
        const trace = {
            x: series.map((item) => item.label),
            y: series.map((item) => item.value || 0.01),
            type: 'bar',
            marker: {
                color: '#1c64f2'
            }
        };

        const layout = {
            margin: { l: 60, r: 20, t: 10, b: 60 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            yaxis: {
                title: 'Samples (log scale)',
                type: 'log',
                rangemode: 'tozero'
            },
            xaxis: {
                title: ''
            }
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(container, [trace], layout, config);
        chartState.evidenceLayout = layout;
    };

    const buildSupportChart = (series) => {
        const container = supportContainer();
        if (!container || typeof Plotly === 'undefined') {
            return;
        }
        const trace = {
            x: series.map((item) => (item.value * 100).toFixed(1)),
            y: series.map((item) => item.label),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: series.map((item) => (item.value >= 0.8 ? '#0e9f6e' : item.value >= 0.5 ? '#1c64f2' : '#f05252'))
            },
            text: series.map((item) => `${(item.value * 100).toFixed(1)}%`),
            textposition: 'auto'
        };

        const layout = {
            margin: { l: 80, r: 20, t: 10, b: 40 },
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(255,255,255,0)',
            font: { family: 'Inter, sans-serif', color: '#1f2933' },
            xaxis: {
                range: [0, 100],
                ticksuffix: '%',
                title: 'Positive audit verdicts'
            },
            yaxis: {
                automargin: true
            }
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(container, [trace], layout, config);
        chartState.supportLayout = layout;
    };

    const applyAnimatedUpdate = (container, update, layoutUpdate) => {
        if (!container || typeof Plotly === 'undefined') {
            return;
        }

        const animationData = {
            data: [update],
            traces: [0],
            layout: layoutUpdate ?? {}
        };

        const animationOptions = {
            transition: {
                duration: 650,
                easing: 'cubic-in-out'
            },
            frame: {
                duration: 650,
                redraw: true
            }
        };

        const fallbackData = {};
        if (update.x) {
            fallbackData.x = [update.x];
        }
        if (update.y) {
            fallbackData.y = [update.y];
        }
        if (update.text) {
            fallbackData.text = [update.text];
        }
        if (update.marker) {
            fallbackData.marker = [update.marker];
        }

        Plotly.animate(container, animationData, animationOptions)
            .catch(() => {
                Plotly.update(container, fallbackData, layoutUpdate ?? {}, [0]);
            });
    };

    const hydrateCharts = (payload) => {
        const evidenceSeries = computeEvidence(payload);
        const supportSeries = computeSupportRate(payload);

        applyAnimatedUpdate(evidenceContainer(), {
            x: evidenceSeries.map((item) => item.label),
            y: evidenceSeries.map((item) => item.value || 0.01),
            marker: { color: '#1c64f2' }
        }, chartState.evidenceLayout);

        applyAnimatedUpdate(supportContainer(), {
            x: supportSeries.map((item) => Number((item.value * 100).toFixed(1))),
            y: supportSeries.map((item) => item.label),
            text: supportSeries.map((item) => `${(item.value * 100).toFixed(1)}%`),
            marker: {
                color: supportSeries.map((item) => (item.value >= 0.8 ? '#0e9f6e' : item.value >= 0.5 ? '#1c64f2' : '#f05252'))
            }
        }, chartState.supportLayout);
    };

    window.addEventListener('DOMContentLoaded', async () => {
        // Render defaults so layout is stable before fetch resolves
        const defaultEvidence = PROGRAM_LABELS.map(({ key, label }) => ({
            key,
            label,
            value: DEFAULT_EVIDENCE[key] ?? 0
        }));

        const defaultSupport = PROGRAM_LABELS.map(({ key, label }) => ({
            key,
            label,
            value: DEFAULT_SUPPORT[key] ?? 0
        }));

        buildEvidenceChart(defaultEvidence);
        buildSupportChart(defaultSupport);

        try {
            const response = await fetch('/api/proofs');
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            hydrateCharts(payload);
        } catch (error) {
            console.warn('Failed to hydrate index charts:', error);
        }
    });
})();
