import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

import {
  RecognitionResponse,
  FileListResponse,
  SaveDigitRequest,
  DeleteFileRequest,
  Dataset,
} from '../../features/digit-recognition/models/recognition.model';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly apiUrl = environment.apiUrl;
  private readonly http = inject(HttpClient);

  /** Получить список файлов для датасета */
  getFiles(dataset: Dataset): Observable<string[]> {
    return this.http
      .get<FileListResponse>(`${this.apiUrl}/test/list`, {
        params: { dataset },
      })
      .pipe(
        map((res) => res.files),
        catchError(this.handleError),
      );
  }

  /** Распознать цифры на изображении */
  recognize(filename: string, dataset: Dataset): Observable<RecognitionResponse> {
    return this.http
      .get<RecognitionResponse>(`${this.apiUrl}/test`, {
        params: { filename, dataset },
      })
      .pipe(catchError(this.handleError));
  }

  /** Сохранить исправленную цифру */
  saveDigit(data: SaveDigitRequest): Observable<{ success: boolean }> {
    return this.http
      .post<{ success: boolean }>(`${this.apiUrl}/save-digit`, data)
      .pipe(catchError(this.handleError));
  }

  /** Удалить файл */
  deleteFile(data: DeleteFileRequest): Observable<{ success: boolean }> {
    return this.http
      .post<{ success: boolean }>(`${this.apiUrl}/delete-file`, data)
      .pipe(catchError(this.handleError));
  }

  private handleError(error: HttpErrorResponse) {
    const message =
      error.error instanceof ErrorEvent
        ? `Client error: ${error.error.message}`
        : `Server error: ${error.status} - ${error.message}`;

    console.error('API Error:', message);
    return throwError(() => new Error(message));
  }
}
