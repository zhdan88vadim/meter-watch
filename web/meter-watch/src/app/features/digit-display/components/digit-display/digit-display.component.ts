import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';

import { ProgressBarModule } from 'primeng/progressbar';
import { CardModule } from 'primeng/card';
import { BadgeModule } from 'primeng/badge';

import { calculateProgress } from '../../models/digit-reading.model';
import { DigitStreamService } from '../../services/digit-stream.service';
import { PersonStatusCardComponent } from '../../../person-detector/components/person-status-card/person-status-card.component';
import { StatusService } from '../../../person-detector/services/status.service';
import { ClockService } from '../../../../core/services/clock.service';
import { formatRemaining } from '../../../../shared/utils/time.util';

@Component({
  selector: 'app-digit-display',
  standalone: true,
  imports: [
    CommonModule,
    ProgressBarModule,
    CardModule,
    BadgeModule,
    PersonStatusCardComponent,
  ],
  templateUrl: './digit-display.component.html',
  styleUrl: './digit-display.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DigitDisplayComponent {
  private readonly digitService = inject(DigitStreamService);
  private readonly statusService = inject(StatusService);
  private readonly clock = inject(ClockService);

  readonly lastActivity = toSignal(this.digitService.pollLastActivity(), {
    initialValue: null,
  });

  readonly status = toSignal(this.statusService.status$, { initialValue: null });

  readonly latestReading = computed(() => this.lastActivity()?.latest_update ?? null);
  readonly history = computed(() => this.lastActivity()?.recent_history ?? []);
  readonly currentDigits = computed<readonly string[]>(
    () => this.latestReading()?.digits ?? [],
  );

  readonly progress = computed(() => {
    const reading = this.latestReading();
    if (!reading) {
      return { timeRemaining: 0, progressPercent: 100, isExpired: true };
    }
  // Reading changes rarely, but clock — every second,
  // so progress is recalculated once per second.
    return calculateProgress(reading.timestamp, this.clock.now());
  });

  readonly timeUntilExpiry = computed(() =>
    formatRemaining(this.progress().timeRemaining),
  );
}
