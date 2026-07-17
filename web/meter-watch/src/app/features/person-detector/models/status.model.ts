// src/app/models/status.model.ts

export interface AlertStatus {
  active: boolean;
  cooldown: boolean;
}

export interface GasStatus {
  flowing: boolean;
  status: '0' | '1' | string; // Если бэкенд может отдавать строкой
}

export interface PersonStatus {
  is_active: boolean;
  last_seen: number; // Timestamp в секундах (float)
  last_seen_str: string; // Строковое представление времени
}

export interface SystemStatus {
  startup_mode: boolean;
}

export interface FullApiResponse {
  alert: AlertStatus;
  gas: GasStatus;
  person: PersonStatus;
  system: SystemStatus;
  timestamp: string; // ISO 8601 строка
}