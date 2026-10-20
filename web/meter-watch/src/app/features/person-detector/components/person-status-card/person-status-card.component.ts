import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { TooltipModule } from 'primeng/tooltip';

import { ClockService } from '../../../../core/services/clock.service';
import { formatAgo } from '../../../../shared/utils/time.util';

export type PersonState = 'present' | 'absent' | 'critical' | 'startup-mode';

@Component({
  selector: 'app-person-status-card',
  standalone: true,
  imports: [CommonModule, TooltipModule],
  templateUrl: './person-status-card.component.html',
  styleUrl: './person-status-card.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PersonStatusCardComponent {
  private readonly clock = inject(ClockService);

  readonly lastSeenTimestamp = input<number | null>(null);
  readonly isPresentNow = input(false);
  readonly isStartupMode = input(false);
  readonly isAlertActive = input(false);
  readonly alertMessage = input('⚠️ Sent to Telegram');

  readonly secondsAgo = computed(() => {
    const ts = this.lastSeenTimestamp();
    if (ts == null) return 0;
    return Math.max(0, Math.floor(this.clock.now() / 1000 - ts));
  });

  readonly status = computed<PersonState>(() => {
    if (this.isStartupMode()) return 'startup-mode';
    if (this.isAlertActive()) return 'critical';
    if (this.isPresentNow()) return 'present';
    return 'absent';
  });

  readonly displayTime = computed(() =>
    formatAgo(this.secondsAgo(), this.isPresentNow()),
  );
}
