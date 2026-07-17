// src/app/services/status.service.ts

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, ReplaySubject, timer } from 'rxjs';
import { catchError, map, retry, switchMap, shareReplay, tap } from 'rxjs/operators';
import { FullApiResponse, PersonStatus, GasStatus, AlertStatus } from '../models/status.model';

@Injectable({
  providedIn: 'root'
})
export class StatusService {
  // ВАЖНО: Замените этот URL на реальный IP вашего бэкенда.
  // Так как вы используете Angular, CORS может быть проблемой. 
  // Если CORS заблокирован на бэкенде, используйте прокси (см. примечание ниже)
  private apiUrl = 'http://192.168.0.254:5000/api/status';

  // Мы используем ReplaySubject(1), чтобы хранить последнее значение и отдавать его новым подписчикам
  // без повторного HTTP-запроса.
  private statusSubject = new ReplaySubject<FullApiResponse>(1);
  public status$ = this.statusSubject.asObservable();

  constructor(private http: HttpClient) {
    // Запускаем опрос сразу при создании сервиса
    this.startPolling();
  }

  /**
   * Метод для однократного получения статуса (если нужно вручную обновить)
   */
  public fetchCurrentStatus(): Observable<FullApiResponse> {
    return this.http.get<FullApiResponse>(this.apiUrl).pipe(
      tap(data => {
        // Сохраняем в кеш при успешной загрузке
        this.statusSubject.next(data);
        console.log(123);
        
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

  // --- Хелперы для быстрого доступа к определенным свойствам ---

  public getPersonStatus(): Observable<PersonStatus> {
    return this.status$.pipe(map(data => data.person));
  }

  public getGasStatus(): Observable<GasStatus> {
    return this.status$.pipe(map(data => data.gas));
  }

  public getAlertStatus(): Observable<AlertStatus> {
    return this.status$.pipe(map(data => data.alert));
  }

  public getTimestamp(): Observable<string> {
    return this.status$.pipe(map(data => data.timestamp));
  }
}