export interface AlertStatus {
  readonly active: boolean;
  readonly cooldown: boolean;
}

export interface GasStatus {
  readonly flowing: boolean;
}

export interface PersonStatus {
  readonly is_active: boolean;
  readonly last_seen: number;      // Unix seconds (float)
  readonly last_seen_str: string;  // human-readable
}

export interface SystemStatus {
  readonly startup_mode: boolean;
}

export interface FullApiResponse {
  readonly alert: AlertStatus;
  readonly gas: GasStatus;
  readonly person: PersonStatus;
  readonly system: SystemStatus;
  readonly timestamp: string;      // ISO 8601
}
