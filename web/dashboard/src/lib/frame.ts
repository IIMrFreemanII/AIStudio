/**
 * Coalesces calls into one per animation frame: however many records arrive between frames, `flush` runs once.
 * `schedule` is injectable for tests.
 */
export class FrameBatcher {
  private pending = false;

  constructor(
    private readonly flush: () => void,
    private readonly schedule: (cb: () => void) => void = defaultSchedule,
  ) {}

  request(): void {
    if (this.pending) return;
    this.pending = true;
    this.schedule(() => {
      this.pending = false;
      this.flush();
    });
  }
}

/** Next animation frame; while the page is hidden (background tab, collapsed pane) browsers pause animation frames,
 * so fall back to a slow timer there and the view is current when it comes back. */
function defaultSchedule(cb: () => void): void {
  if (typeof document !== 'undefined' && document.hidden) setTimeout(cb, 250);
  else requestAnimationFrame(() => cb());
}
