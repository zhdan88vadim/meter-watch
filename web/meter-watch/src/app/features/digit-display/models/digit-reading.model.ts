export interface LastActivity {
  readonly recent_history: readonly DigitReading[];
  readonly latest_update: DigitReading;
}

export interface DigitReading {
  readonly digits: readonly string[];
  readonly time: string;
  readonly timestamp: number;   // Unix seconds
}

export interface ProgressState {
  readonly timeRemaining: number;   // seconds
  readonly progressPercent: number; // 0..100
  readonly isExpired: boolean;
}

export const COUNTDOWN_DURATION_MS = 5 * 60 * 1000;

export function calculateProgress(lastTimestamp: number, nowMs = Date.now()): ProgressState {
  const elapsed = nowMs - lastTimestamp * 1000;
  const remaining = Math.max(0, COUNTDOWN_DURATION_MS - elapsed);
  const percent = (remaining / COUNTDOWN_DURATION_MS) * 100;

  return {
    timeRemaining: Math.floor(remaining / 1000),
    progressPercent: Math.round(percent),
    isExpired: elapsed >= COUNTDOWN_DURATION_MS,
  };
}
