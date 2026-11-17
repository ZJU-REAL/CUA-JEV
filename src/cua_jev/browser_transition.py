"""Bounded passive DOM/animation settling before one post-action screenshot.

Observed quiet is not application readiness or a substitute for input preflight.
No screenshots, benchmark state, network-idle waits, or input are used here.
"""

SETTLE_TRANSITION = r"""() => new Promise(resolve => {
    const started = performance.now(), budget = 600, quiet = 50;
    let changed = started, samples = 0, finished = false, frame;
    const observer = new MutationObserver(() => { changed = performance.now(); });
    observer.observe(document.documentElement, {
        subtree: true, childList: true, characterData: true, attributes: true
    });
    const onScroll = () => { changed = performance.now(); };
    document.addEventListener('scroll', onScroll, true);
    const finish = status => {
        if (finished) return;
        finished = true;
        observer.disconnect();
        document.removeEventListener('scroll', onScroll, true);
        cancelAnimationFrame(frame);
        clearTimeout(watchdog);
        resolve({status, elapsed_ms: performance.now() - started, samples,
                 future_changes_excluded: false});
    };
    const watchdog = setTimeout(() => finish('budget_exhausted'), budget);
    const sample = () => {
        if (finished) return;
        ++samples;
        const now = performance.now();
        // Infinite decorative animations must not force every action to wait
        // for the entire budget. They remain visible to ordinary preflight.
        const active = document.getAnimations().some(animation => {
            const timing = animation.effect?.getComputedTiming();
            return animation.playState === 'running' && timing &&
                   Number.isFinite(timing.endTime);
        });
        if (active) changed = now;
        if (!active && now - changed >= quiet) finish('observed_quiet');
        else frame = requestAnimationFrame(sample);
    };
    frame = requestAnimationFrame(sample);
})"""
