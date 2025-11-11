(() => {
    const supportsFinePointer = window.matchMedia('(pointer: fine)').matches;

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

    const numberFormatter = (decimals = 0) => new Intl.NumberFormat('en-US', {
        maximumFractionDigits: decimals,
        minimumFractionDigits: decimals
    });

    const formatStatValue = (el, rawValue, options = {}) => {
        const format = options.format || el.dataset.animate || 'text';
        const decimals = typeof options.decimals === 'number'
            ? options.decimals
            : Number(el.dataset.decimals ?? (format === 'percent' ? 1 : 0));
        const formatter = numberFormatter(decimals);

        if (format === 'percent') {
            const percentValue = (rawValue ?? 0) * 100;
            const text = `${formatter.format(percentValue)}%`;
            el.textContent = text;
            return text;
        }

        if (format === 'ratio') {
            const denominator = options.denominator
                ?? Number(el.dataset.denominator ?? 0);
            const numeratorText = formatter.format(rawValue ?? 0);
            const ratioText = denominator
                ? `${numeratorText} / ${formatter.format(denominator)}`
                : numeratorText;
            el.textContent = ratioText;
            return ratioText;
        }

        if (format === 'template') {
            const template = options.template || el.dataset.template || '{value}';
            const text = template.replace('{value}', formatter.format(rawValue ?? 0));
            el.innerHTML = text;
            return text;
        }

        if (format === 'number') {
            const text = formatter.format(rawValue ?? 0);
            el.textContent = text;
            return text;
        }

        const fallback = rawValue == null ? '' : String(rawValue);
        el.textContent = fallback;
        return fallback;
    };

    const animateStatValue = (el, target, options = {}) => {
        if (!el) {
            return;
        }

        const format = options.format || el.dataset.animate || 'text';
        const decimals = typeof options.decimals === 'number'
            ? options.decimals
            : Number(el.dataset.decimals ?? (format === 'percent' ? 1 : 0));
        const denominator = options.denominator ?? (el.dataset.denominator ? Number(el.dataset.denominator) : undefined);
        const template = options.template || el.dataset.template;

        const destination = typeof target === 'number'
            ? target
            : Number.parseFloat(target);

        if (!Number.isFinite(destination)) {
            formatStatValue(el, target, { format, decimals, denominator, template });
            return;
        }

        const startValue = el.dataset.currentValue ? Number(el.dataset.currentValue) : Number(el.dataset.value ?? destination);
        const easing = (t) => 1 - Math.pow(1 - t, 3);
        const duration = clamp(Number(options.duration ?? 900), 320, 2000);
        const startTime = performance.now();

        const step = (now) => {
            const elapsed = now - startTime;
            const progress = elapsed >= duration ? 1 : clamp(elapsed / duration, 0, 1);
            const eased = easing(progress);
            const currentValue = startValue + (destination - startValue) * eased;

            formatStatValue(el, currentValue, { format, decimals, denominator, template });

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                formatStatValue(el, destination, { format, decimals, denominator, template });
                el.dataset.currentValue = String(destination);
                el.dataset.value = String(destination);
                if (typeof denominator === 'number') {
                    el.dataset.denominator = String(denominator);
                }
            }
        };

        requestAnimationFrame(step);
    };

    const initAnimatedStats = () => {
        document.querySelectorAll('[data-animate]').forEach((el) => {
            const stored = el.dataset.value ? Number(el.dataset.value) : undefined;
            if (Number.isFinite(stored)) {
                el.dataset.currentValue = String(stored);
                formatStatValue(el, stored);
            }
        });
    };

    const initTilt = () => {
        if (!supportsFinePointer) {
            return;
        }

        const tiltTargets = document.querySelectorAll('[data-tilt]');
        tiltTargets.forEach((el) => {
            const maxTilt = Number.parseFloat(el.dataset.tiltMax ?? (el.classList.contains('hero') ? 8 : 5));

            const handleMove = (event) => {
                const rect = el.getBoundingClientRect();
                const relativeX = (event.clientX - rect.left) / rect.width;
                const relativeY = (event.clientY - rect.top) / rect.height;
                const tiltX = clamp(0.5 - relativeX, -0.5, 0.5) * maxTilt;
                const tiltY = clamp(relativeY - 0.5, -0.5, 0.5) * maxTilt;

                el.style.setProperty('--tilt-x', `${tiltX.toFixed(3)}deg`);
                el.style.setProperty('--tilt-y', `${tiltY.toFixed(3)}deg`);
                el.dataset.tiltActive = 'true';
            };

            const resetTilt = () => {
                el.style.setProperty('--tilt-x', '0deg');
                el.style.setProperty('--tilt-y', '0deg');
                delete el.dataset.tiltActive;
            };

            el.addEventListener('pointermove', handleMove);
            el.addEventListener('pointerleave', resetTilt);
        });
    };

    const initTimeline = () => {
        const timelines = document.querySelectorAll('[data-timeline]');
        if (!timelines.length) {
            return;
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) {
                    return;
                }
                const item = entry.target;
                item.classList.add('is-visible');
                observer.unobserve(item);

                const timeline = item.closest('[data-timeline]');
                if (!timeline) {
                    return;
                }
                const visibleItems = timeline.querySelectorAll('.timeline-item.is-visible');
                const lastVisible = visibleItems[visibleItems.length - 1];
                if (lastVisible) {
                    const offset = lastVisible.offsetTop + lastVisible.offsetHeight;
                    timeline.style.setProperty('--timeline-progress', `${offset}px`);
                }
            });
        }, {
            threshold: 0.4
        });

        timelines.forEach((timeline) => {
            timeline.style.setProperty('--timeline-progress', '0px');
            timeline.querySelectorAll('.timeline-item').forEach((item) => observer.observe(item));
        });
    };

    const initCardLinks = () => {
        document.querySelectorAll('[data-card-link]').forEach((card) => {
            const href = card.getAttribute('data-card-link');
            if (!href) {
                return;
            }

            const handleActivate = () => window.location.assign(href);

            card.addEventListener('click', (event) => {
                if ((event.target instanceof Element) && event.target.closest('a')) {
                    return;
                }
                handleActivate();
            });

            card.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleActivate();
                }
            });
        });
    };

    window.addEventListener('DOMContentLoaded', () => {
        initAnimatedStats();
        initTilt();
        initTimeline();
        initCardLinks();
    });

    window.animateStatValue = (el, value, options = {}) => animateStatValue(el, value, options);
})();
