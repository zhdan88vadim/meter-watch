import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, ReplaySubject, timer } from 'rxjs';
import { catchError, retry, switchMap, shareReplay, tap } from 'rxjs/operators';
import { FullApiResponse } from '../models/status.model';
import { environment } from '../../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class StatusService {
  private readonly apiUrl = environment.personDetectApiUrl;  
  private statusSubject = new ReplaySubject<FullApiResponse>(1);
  public status$ = this.statusSubject.asObservable();

  constructor(private http: HttpClient) {
    this.startPolling();
  }

  public fetchCurrentStatus(): Observable<FullApiResponse> {
    return this.http.get<FullApiResponse>(`${this.apiUrl}/status`).pipe(
      tap(data => {
        this.statusSubject.next(data);      
      }),
      catchError((error) => {
        console.error('Ошибка при получении статуса с API:', error);
        // Возвращаем ошибку, чтобы подписчик знал о проблеме
        throw error; 
      })
    );
  }

  /**
   * Автоматический опрос (Polling) каждые 3 секунды
   */
  private startPolling() {
    timer(0, 3000) // 0 - начать сразу, 3000 - повторять каждые 3 секунды
      .pipe(
        switchMap(() => this.fetchCurrentStatus()),
        retry(3), // Если ошибка - попробовать переподключиться 3 раза
        shareReplay(1)
      )
      .subscribe({
        next: (data) => {
          // Данные уже сохранены в Subject внутри fetchCurrentStatus
        },
        error: (err) => {
          console.warn('Не удается подключиться к API. Проверьте соединение с 192.168.0.254');
        }
      });
  }

}