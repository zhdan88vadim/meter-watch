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
  SaveResult,
} from '../../features/digit-recognition/models/recognition.model';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly apiUrl = environment.apiUrl;
  private readonly http = inject(HttpClient);

  getFiles(dataset: Dataset): Observable<readonly string[]> {
    return this.http
      .get<FileListResponse>(`${this.apiUrl}/test/list`, { params: { dataset } })
      .pipe(
        map((res) => res.files ?? []),
        catchError(this.handleError),
      );
  }

  recognize(filename: string, dataset: Dataset): Observable<RecognitionResponse> {
    return this.http
      .get<RecognitionResponse>(`${this.apiUrl}/test`, {
        params: { filename, dataset },
      })
      .pipe(catchError(this.handleError));
  }

  saveDigit(data: SaveDigitRequest): Observable<SaveResult> {
    return this.http
      .post<SaveResult>(`${this.apiUrl}/save-digit`, data)
      .pipe(catchError(this.handleError));
  }

  deleteFile(data: DeleteFileRequest): Observable<SaveResult> {
    return this.http
      .post<SaveResult>(`${this.apiUrl}/delete-file`, data)
      .pipe(catchError(this.handleError));
  }

  private readonly handleError = (error: HttpErrorResponse) => {
    const message =
      error.error instanceof ErrorEvent
        ? `Client error: ${error.error.message}`
        : `Server error: ${error.status} — ${error.message}`;
    console.error('API Error:', message);
    return throwError(() => new Error(message));
  };
}