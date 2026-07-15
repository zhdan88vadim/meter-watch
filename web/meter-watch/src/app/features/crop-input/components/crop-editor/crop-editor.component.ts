import { Component, DestroyRef, inject, ChangeDetectorRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { ButtonModule } from 'primeng/button';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { ToastModule } from 'primeng/toast';
import { MessageService } from 'primeng/api';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { CropService } from '../../services/crop.service';
import { CropParams, CropResult } from '../../models/crop.models';
import { CropControlsComponent } from '../crop-controls/crop-controls.component';
import { CropPreviewComponent } from '../crop-preview/crop-preview.component';
import { ImageSourceComponent } from '../image-source/image-source.component';
import { ResultsDisplayComponent } from '../results-display/results-display.component';
import { testImageBase64 } from './test-image.constants';

@Component({
  selector: 'app-crop-editor',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    ButtonModule,
    ProgressSpinnerModule,
    ToastModule,
    CropControlsComponent,
    ImageSourceComponent,
    ResultsDisplayComponent,
    CropPreviewComponent,
  ],
  providers: [MessageService, CropService],
  templateUrl: './crop-editor.component.html',
  styleUrls: ['./crop-editor.component.scss'],
})
export class CropEditorComponent {
  @ViewChild(CropPreviewComponent) previewComponent!: CropPreviewComponent;

  private destroyRef = inject(DestroyRef);
  private cdr = inject(ChangeDetectorRef);

  currentImageBase64: string | null = null;
  currentCropParams: CropParams = {
    crop_top: 45,
    crop_bottom: 35,
    crop_left: 8,
    crop_right: 0,
  };
  cropResult: CropResult | null = null;
  loading = false;
  imageWidth = 0;
  imageHeight = 0;
  maxTop = 300;
  maxBottom = 300;
  maxLeft = 300;
  maxRight = 300;

  constructor(
    private cropService: CropService,
    private messageService: MessageService,
  ) {
    this.loadCurrentConfig().then(() => this.onLoadTestImageClick());
  }

  async loadCurrentConfig(): Promise<void> {
    try {
      const data = await this.cropService
        .getConfig()
        .pipe(takeUntilDestroyed(this.destroyRef))
        .toPromise();
      if (data?.status === 'success' && data.config) {
        this.currentCropParams = { ...data.config };
      }
    } catch (error) {
      console.error('Error loading config:', error);
    }
  }

  onCameraClick(): void {
    this.fetchFromCamera();
  }

  onLoadTestImageClick(): void {
    this.fetchFromTestImage();
  }

  onFileSelected(file: File): void {
    const reader = new FileReader();
    reader.onload = (e: any) => {
      this.currentImageBase64 = e.target.result.split(',')[1];
      this.previewCrop();
    };
    reader.readAsDataURL(file);
  }

  async fetchFromTestImage(): Promise<void> {
    this.loading = true;
    try {
      const data = await this.cropService
        .previewCrop(this.currentCropParams, false, testImageBase64)
        .pipe(takeUntilDestroyed(this.destroyRef))
        .toPromise();
      if (data?.status === 'success' && data.result) {
        this.displayResults(data.result);
        this.currentImageBase64 = null;
      } else {
        this.showError(data?.error || 'Ошибка получения изображения');
      }
    } catch (error) {
      this.showError('Ошибка получения изображения с камеры');
    } finally {
      this.loading = false;
    }
  }

  async fetchFromCamera(): Promise<void> {
    this.loading = true;
    try {
      const data = await this.cropService
        .previewCrop(this.currentCropParams, true)
        .pipe(takeUntilDestroyed(this.destroyRef))
        .toPromise();
      if (data?.status === 'success' && data.result) {
        this.displayResults(data.result);
        this.currentImageBase64 = null;
      } else {
        this.showError(data?.error || 'Ошибка получения изображения');
      }
    } catch (error) {
      this.showError('Ошибка получения изображения с камеры');
    } finally {
      this.loading = false;
    }
  }

  async previewCrop(): Promise<void> {
    this.loading = true;
    try {
      const useCamera = this.currentImageBase64 === null;
      const data = await this.cropService
        .previewCrop(this.currentCropParams, useCamera, this.currentImageBase64 || undefined)
        .pipe(takeUntilDestroyed(this.destroyRef))
        .toPromise();
      if (data?.status === 'success' && data.result) {
        this.displayResults(data.result);
      } else {
        this.showError(data?.error || 'Ошибка обработки изображения');
      }
    } catch (error) {
      this.showError('Ошибка обработки изображения');
    } finally {
      this.loading = false;
    }
  }

  async saveToConfig(): Promise<void> {
    this.loading = true;
    try {
      const data = await this.cropService
        .updateConfig(this.currentCropParams)
        .pipe(takeUntilDestroyed(this.destroyRef))
        .toPromise();
      if (data?.status === 'success') {
        this.showSuccess('Параметры успешно сохранены в конфиг!');
        await this.loadCurrentConfig();
      } else {
        this.showError(data?.message || 'Ошибка сохранения');
      }
    } catch (error) {
      this.showError('Ошибка сохранения параметров');
    } finally {
      this.loading = false;
    }
  }

  resetToDefault(): void {
    this.currentCropParams = {
      crop_top: 45,
      crop_bottom: 35,
      crop_left: 0,
      crop_right: 20,
    };
    this.updateMaxValues();
    this.showSuccess('Параметры сброшены к значениям по умолчанию');
    this.previewCrop();
  }

  onParamChange(event: { param: keyof CropParams; value: number }): void {
    this.currentCropParams[event.param] = event.value;
    this.currentCropParams = { ...this.currentCropParams };
    this.updateMaxValues();
  }

  updateCropParams(params: CropParams): void {
    this.currentCropParams = { ...params };
  }

  private updateMaxValues(): void {
    if (this.imageHeight > 0) {
      this.maxTop = Math.max(0, this.imageHeight - 10);
      this.maxBottom = Math.max(0, this.imageHeight - 10);
    }
    if (this.imageWidth > 0) {
      this.maxLeft = Math.max(0, this.imageWidth - 10);
      this.maxRight = Math.max(0, this.imageWidth - 10);
    }
  }

  private displayResults(result: CropResult): void {
    this.cropResult = result;
    if (result.original_image) {
      this.loadOriginalImage(result.original_image);
    }
  }

  private loadOriginalImage(imageBase64: string): void {
    const img = new Image();
    img.onload = () => {
      this.imageWidth = img.width;
      this.imageHeight = img.height;
      this.updateMaxValues();
      this.cdr.detectChanges();
    };
    img.src = `data:image/png;base64,${imageBase64}`;
  }

  private showSuccess(message: string): void {
    this.messageService.add({ severity: 'success', summary: 'Успех', detail: message });
  }

  private showError(message: string): void {
    this.messageService.add({ severity: 'error', summary: 'Ошибка', detail: message });
  }
}
