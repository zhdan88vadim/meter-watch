import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { FileUploadModule } from 'primeng/fileupload';

@Component({
  selector: 'app-image-source',
  standalone: true,
  imports: [CommonModule, ButtonModule, FileUploadModule],
  template: `
    <div class="control-group">
      <h3>📷 Источник изображения</h3>
      <div class="button-group">
        <p-button label="📸 Получить с камеры" (onClick)="onCameraClick()" styleClass="p-button-primary"></p-button>
        <p-button label="Load test image" (onClick)="onLoadTestImageClick()" styleClass="p-button-primary"></p-button>
        <p-fileUpload mode="basic" [multiple]="false" accept="image/*" 
                      (onSelect)="onFileSelected($event)" chooseLabel="📁 Загрузить файл"></p-fileUpload>
      </div>
    </div>
  `,
  styles: [`
    .control-group {
      background: #f8f9fa;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .control-group h3 {
      color: #333;
      margin-bottom: 15px;
      font-size: 1.2em;
    }
    .button-group {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
  `]
})
export class ImageSourceComponent {
  @Output() cameraClick = new EventEmitter<void>();
  @Output() loadTestImageClick = new EventEmitter<void>();
  @Output() fileSelected = new EventEmitter<File>();

  onCameraClick(): void {
    this.cameraClick.emit();
  }

  onLoadTestImageClick(): void {
    this.loadTestImageClick.emit();
  }

  onFileSelected(event: any): void {
    if (event.files && event.files[0]) {
      this.fileSelected.emit(event.files[0]);
    }
  }
}