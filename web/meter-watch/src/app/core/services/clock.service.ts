import { Injectable, DestroyRef, inject, signal } from '@angular/core';

/**
 * Single source of "current time" (updated once per second).
 * All components that need a "tick" for countdown
 * depend on this service — they don't spawn their own timers.
 */
@Injectable({ providedIn: 'root' })
export class ClockService {
  private readonly destroyRef = inject(DestroyRef);
  private readonly _now = signal<number>(Date.now());

  /** Current time (ms, Unix epoch). Changes every second. */
  readonly now = this._now.asReadonly();

  constructor() {
    const handle = setInterval(() => this._now.set(Date.now()), 1000);
    this.destroyRef.onDestroy(() => clearInterval(handle));
  }
}