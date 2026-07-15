import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CropResult } from '../../models/crop.models';
import { DigitCardComponent } from '../digit-card/digit-card.component';

@Component({
  selector: 'app-results-display',
  standalone: true,
  imports: [CommonModule, DigitCardComponent],
  template: `
    @if (result) {
      <div class="image-container">
        <h3>✂️ Обрезанное изображение</h3>
        <img [src]="'data:image/png;base64,' + result.cropped_image" alt="Обрезанное изображение" />
        @if (result.original_size && result.cropped_size) {
          <div class="info-text">
            Оригинал: {{ result.original_size.width }}x{{ result.original_size.height }}px |
            Обрезано: {{ result.cropped_size.width }}x{{ result.cropped_size.height }}px
          </div>
        }
      </div>

      <div class="image-container">
        <h3>🔍 Пороговое изображение</h3>
        <img
          [src]="'data:image/png;base64,' + result.threshold_image"
          alt="Пороговое изображение"
        />
      </div>

      <div class="image-container">
        <h3>🔢 Распознанные цифры</h3>
        <div class="digits-grid">
          @for (digit of result.digits; track digit.position) {
            <app-digit-card [digit]="digit"></app-digit-card>
          }
        </div>
        @if (result.digits.length === 0) {
          <div class="alert alert-error">Цифры не найдены</div>
        }
        <div class="result-number">Распознанное число: {{ result.full_number || '---' }}</div>
      </div>
    }
  `,
  styles: [
    `
      .image-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
      }
      .image-container h3 {
        margin-bottom: 15px;
        color: #333;
      }
      .image-container img {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
      }
      .info-text {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
      }
      .digits-grid {
        display: flex;
        gap: 16px;
        justify-content: center;
        margin-bottom: 16px;
      }
      .alert {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
      }
      .alert-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
      }
      .result-number {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-top: 20px;
      }
    `,
  ],
})
export class ResultsDisplayComponent {
  @Input() result: CropResult | null = null;
}
