import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { TagModule } from 'primeng/tag';
import { ProgressBarModule } from 'primeng/progressbar';
import { TooltipModule } from 'primeng/tooltip';

import { DigitResult } from '../../models/recognition.model';

type ProgressSeverity = 'success' | 'warn' | 'danger';

@Component({
  selector: 'app-digit-card',
  standalone: true,
  imports: [
    CommonModule,
    CardModule,
    ButtonModule,
    TagModule,
    ProgressBarModule,
    TooltipModule,
  ],
  templateUrl: './digit-card.html',
  styleUrl: './digit-card.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DigitCardComponent {
  readonly digit = input.required<DigitResult>();
  readonly position = input.required<number>();
  readonly selectedDigit = input<number>();

  readonly digitSelected = output<{ position: number; digit: number }>();
  readonly digitDeselected = output<number>();

  readonly confidencePercent = computed(
    () => Math.round(this.digit().confidence * 1000) / 10,
  );

  readonly progressSeverity = computed<ProgressSeverity>(() => {
    const c = this.digit().confidence;
    if (c >= 0.9) return 'success';
    if (c >= 0.7) return 'warn';
    return 'danger';
  });

  readonly availableDigits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

  isSelected(d: number): boolean {
    return this.selectedDigit() === d;
  }

  onDigitClick(digit: number): void {
    if (this.isSelected(digit)) {
      this.digitDeselected.emit(this.position());
    } else {
      this.digitSelected.emit({ position: this.position(), digit });
    }
  }
}
