import { Injectable, inject } from '@angular/core';
import { Observable, interval } from 'rxjs';
import { switchMap, startWith, shareReplay } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';

import { LastActivity } from '../models/digit-reading.model';
import { environment } from '../../../../environments/environment';
import { POLLING_CONFIG } from '../../../core/tokens/polling-config.token';

@Injectable({ providedIn: 'root' })
export class DigitStreamService {
  private readonly http = inject(HttpClient);
  private readonly config = inject(POLLING_CONFIG);
  private readonly apiUrl = environment.apiUrl;

  getLastActivity(): Observable<LastActivity> {
    return this.http.get<LastActivity>(`${this.apiUrl}/last_activity`);
  }

  pollLastActivity(intervalSeconds = this.config.digitPollSeconds): Observable<LastActivity> {
    return interval(intervalSeconds * 1000).pipe(
      startWith(0),
      switchMap(() => this.getLastActivity()),
      shareReplay({ bufferSize: 1, refCount: true }),
    );
  }
}
