import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ProgressBarModule } from 'primeng/progressbar';
import { CardModule } from 'primeng/card';
import { BadgeModule } from 'primeng/badge';
import { Subject, takeUntil } from 'rxjs';
import { DigitReading, calculateProgress } from '../../models/digit-reading.model';
import { DigitStreamService } from '../../services/digit-stream.service';
import { PersonStatusCardComponent } from '../../../person-detector/components/person-status-card/person-status-card.component';
import { StatusService } from '../../../person-detector/services/status.service';

@Component({
  selector: 'app-digit-display',
  standalone: true,
  imports: [CommonModule, ProgressBarModule, CardModule, BadgeModule, PersonStatusCardComponent],
  templateUrl: './digit-display.component.html',
  styleUrls: ['./digit-display.component.scss'],
})
export class DigitDisplayComponent implements OnInit, OnDestroy {
  latestReading: DigitReading | null = null;
  history: DigitReading[] = [];
  progressPercent = 100;
  timeRemaining = 300;
  isExpired = false;

  private destroy$ = new Subject<void>();
  private updateTimer?: any;

  constructor(
    private digitService: DigitStreamService,
    public statusService: StatusService
  ) {

    this.statusService.status$.subscribe(x => {
      console.log(x);
      
    })

  }

  ngOnInit(): void {
    this.digitService
      .pollLastActivity(5)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          this.history = data.recent_history;
          this.latestReading = data.latest_update;
          this.updateProgress();
        },
        error: (err) => console.error('Polling error:', err),
      });

    this.updateTimer = setInterval(() => {
      this.updateProgress();
    }, 1000);
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.updateTimer) clearInterval(this.updateTimer);
  }

  private updateProgress(): void {
    if (!this.latestReading) return;
    const progress = calculateProgress(this.latestReading.timestamp);
    this.progressPercent = progress.progressPercent;
    this.timeRemaining = progress.timeRemaining;
    this.isExpired = progress.isExpired;
  }

  get currentDigits(): string[] {
    return this.latestReading?.digits || [];
  }

  get timeUntilExpiry(): string {
    const mins = Math.floor(this.timeRemaining / 60);
    const secs = this.timeRemaining % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}