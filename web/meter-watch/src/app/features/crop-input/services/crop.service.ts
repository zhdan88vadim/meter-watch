// src/app/services/crop.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiResponse, CropParams, CropResult } from '../models/crop.models';
import { environment } from '../../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class CropService {
  private apiBase = environment.apiUrl

  constructor(private http: HttpClient) {}

  getConfig(): Observable<ApiResponse<any>> {
    return this.http.get<ApiResponse<any>>(`${this.apiBase}/config`);
  }

  updateConfig(params: CropParams): Observable<ApiResponse<any>> {
    return this.http.post<ApiResponse<any>>(`${this.apiBase}/config`, params);
  }

  previewCrop(params: CropParams, useCamera: boolean, imageBase64?: string): Observable<ApiResponse<CropResult>> {
    const payload: any = {
      use_camera: useCamera,
      crop_top: params.crop_top,
      crop_bottom: params.crop_bottom,
      crop_left: params.crop_left,
      crop_right: params.crop_right
    };

    if (!useCamera && imageBase64) {
      payload.use_camera = false;
      payload.image_base64 = imageBase64;
    }

    return this.http.post<ApiResponse<CropResult>>(`${this.apiBase}/preview_crop`, payload);
  }
}