/** Seconds → `M:SS` */
export function formatRemaining(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.max(0, Math.floor(seconds % 60));
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Seconds ago → human-readable string ('Now', 'N sec. ago', 'N min. ago') */
export function formatAgo(seconds: number, present: boolean): string {
  if (present || seconds < 5) return 'Now';
  if (seconds < 60) return `${Math.floor(seconds)} sec. ago`;
  return `${Math.floor(seconds / 60)} min. ago`;
}
