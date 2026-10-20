import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, timer, of } from 'rxjs';
import { catchError, shareReplay, switchMap } from 'rxjs/operators';

import { FullApiResponse } from '../models/status.model';
import { environment } from '../../../../environments/environment';
import { POLLING_CONFIG } from '../../../core/tokens/polling-config.token';

@Injectable({ providedIn: 'root' })
export class StatusService {
  private readonly http = inject(HttpClient);
  private readonly config = inject(POLLING_CONFIG);
  private readonly apiUrl = environment.personDetectApiUrl;

  /**
   * Single status stream. Errors don't kill polling — they just emit null.
   * `refCount: true` — polling stops when there are no subscribers.
   */
  readonly status$: Observable<FullApiResponse | null> = timer(
    0,
    this.config.statusPollMs,
  ).pipe(
    switchMap(() =>
      this.http
        .get<FullApiResponse>(`${this.apiUrl}/status`)
        .pipe(catchError((err) => {
          console.warn('[StatusService] poll failed:', err?.message ?? err);
          return of(null);
        })),
    ),
    shareReplay({ bufferSize: 1, refCount: true }),
  );

  /** One-shot request without side effects. */
  fetchCurrentStatus(): Observable<FullApiResponse> {
    return this.http.get<FullApiResponse>(`${this.apiUrl}/status`);
  }
}
