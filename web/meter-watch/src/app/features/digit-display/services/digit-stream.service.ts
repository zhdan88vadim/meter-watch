// digit-stream.service.ts

import { Injectable } from '@angular/core';
import { Observable, interval, switchMap, startWith, shareReplay } from 'rxjs';
import { DigitReading, LastActivity } from '../models/digit-reading.model';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class DigitStreamService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getLastActivity(): Observable<LastActivity> {
    return this.http.get<LastActivity>(`${this.apiUrl}/last_activity`);
  }

  /**
   * 🔁 Polling - опрашивает сервер каждые N секунд
   */
  pollLastActivity(intervalSeconds: number = 5): Observable<LastActivity> {
    return interval(intervalSeconds * 1000).pipe(
      startWith(0), // Первый запрос сразу
      switchMap(() => this.getLastActivity()),
      shareReplay(1), // Кэширует последний результат для новых подписчиков
    );
  }
}
