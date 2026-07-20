import { Component, input, computed, signal, effect, inject, DestroyRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TooltipModule } from 'primeng/tooltip';

export type PersonState = 'present' | 'absent' | 'critical';

@Component({
  selector: 'app-person-status-card',
  standalone: true,
  imports: [CommonModule, TooltipModule],
  templateUrl: './person-status-card.component.html',
  styleUrl: './person-status-card.component.scss'
})
export class PersonStatusCardComponent {
  private destroyRef = inject(DestroyRef);

  lastSeenTimestamp = input<number | null>(null);
  isPresentNow = input<boolean>(false);
  isStartupMode = input<boolean>(false);
  isAlertActive = input<boolean>(false);
  alertMessage = input<string>('⚠️ Отправлено в Telegram');

  // --- Внутреннее состояние ---
  private secondsAgo = signal(0);
  private timerInterval: any;

  constructor() {
    // Effect сработает при изменении ЛЮБОГО из сигналов, прочитанных внутри него.
    effect(() => {
      // Читаем значения сигналов внутри effect. Это и есть подписка на изменения.
      const timestamp = this.lastSeenTimestamp(); 
      
      console.log('Эффект сработал. Время:', timestamp);

      // Очищаем старый таймер перед созданием нового
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }

      // Если данные есть - запускаем логику
      if (timestamp !== null && timestamp !== undefined) {
        this.updateTimeAgo(timestamp);
        this.timerInterval = setInterval(() => this.updateTimeAgo(timestamp), 1000);
      } else {
        this.secondsAgo.set(0);
      }
    });

    // Очистка таймера при уничтожении компонента
    this.destroyRef.onDestroy(() => {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
      }
    });
  }

  status = computed(() => {
    if (this.isStartupMode()) return 'startup-mode';
    if (this.isPresentNow()) return 'present';
    if (this.isAlertActive()) return 'critical';
    return 'absent';
  });

  // Форматированное время
  displayTime = computed(() => {
    const secs = this.secondsAgo();
    if (this.isPresentNow() || secs < 5) return 'Сейчас';
    
    if (secs < 60) return `${Math.round(secs)} сек. назад`;
    const mins = Math.round(secs / 60);
    return `${mins} мин. назад`;
  });

  private updateTimeAgo(timestamp: number): void {
    const now = Math.floor(Date.now() / 1000);
    this.secondsAgo.set(Math.max(0, now - timestamp));
  }
}