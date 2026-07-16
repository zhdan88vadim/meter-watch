// src/app/features/digit-recognition/components/digit-card/digit-card.component.ts
import { Component, input, output, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { TagModule } from 'primeng/tag';
import { ProgressBarModule } from 'primeng/progressbar';
import { TooltipModule } from 'primeng/tooltip';
import { DigitResult } from '../../models/recognition.model';

@Component({
  selector: 'app-digit-card',
  standalone: true,
  imports: [CommonModule, CardModule, ButtonModule, TagModule, ProgressBarModule, TooltipModule],
  templateUrl: './digit-card.html',
  styleUrl: './digit-card.scss'
})
export class DigitCardComponent {
  readonly digit = input.required<DigitResult>();
  readonly position = input.required<number>(); // 0-based
  readonly selectedDigit = input<number>();
  readonly digitSelected = output<{ position: number; digit: number }>();
  readonly digitDeselected = output<number>();

  // Вычисляем, выбрана ли конкретная цифра
  isSelected(d: number): boolean {
    return this.selectedDigit() === d;
  }

  // Обработчик клика по кнопке цифры
  onDigitClick(digit: number): void {
    if (this.isSelected(digit)) {
      this.digitDeselected.emit(this.position());
    } else {
      this.digitSelected.emit({ position: this.position(), digit });
    }
  }

  // Форматирование уверенности
  get confidencePercent(): string {
    return `${(this.digit().confidence * 100).toFixed(1)}`;
  }

  // Цвет прогресс-бара в зависимости от уверенности
  get progressColor(): string {
    const conf = this.digit().confidence;
    if (conf >= 0.9) return 'success';
    if (conf >= 0.7) return 'warn';
    return 'danger';
  }
}