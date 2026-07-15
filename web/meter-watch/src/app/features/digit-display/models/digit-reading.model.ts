export interface LastActivity {
  recent_history: DigitReading[];
  latest_update: DigitReading;
}

export interface DigitReading {
  digits: string[];
  time: string;
  timestamp: number;
}

export interface DigitDisplayData {
  latestReading: DigitReading | null;
  readings: DigitReading[];
  timeRemaining: number; // Seconds until 5 min expires
  progressPercent: number; // 0-100 for progress bar
  isExpired: boolean; // True if > 5 min since last reading
}

export const COUNTDOWN_DURATION_MS = 5 * 60 * 1000; // 5 minutes

export function calculateProgress(lastTimestamp: number): {
  timeRemaining: number;
  progressPercent: number;
  isExpired: boolean;
} {
  const now = Date.now();
  const elapsed = now - lastTimestamp * 1000;
  const remaining = Math.max(0, COUNTDOWN_DURATION_MS - elapsed);
  const percent = (remaining / COUNTDOWN_DURATION_MS) * 100;

  return {
    timeRemaining: Math.floor(remaining / 1000),
    progressPercent: Math.round(percent),
    isExpired: elapsed >= COUNTDOWN_DURATION_MS,
  };
}
