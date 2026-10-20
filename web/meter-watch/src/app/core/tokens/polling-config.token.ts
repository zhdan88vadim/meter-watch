import { InjectionToken } from '@angular/core';

export interface PollingConfig {
  readonly digitPollSeconds: number;
  readonly statusPollMs: number;
}

export const DEFAULT_POLLING_CONFIG: PollingConfig = {
  digitPollSeconds: 5,
  statusPollMs: 3000,
};

export const POLLING_CONFIG = new InjectionToken<PollingConfig>(
  'POLLING_CONFIG',
  { factory: () => DEFAULT_POLLING_CONFIG },
);
