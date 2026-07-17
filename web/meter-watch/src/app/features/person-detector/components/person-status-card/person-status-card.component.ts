// src/app/features/presence/components/person-status-card/person-status-card.component.ts
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

  // --- ВХОДНЫЕ ДАННЫЕ (ТЕПЕРЬ ЭТО СИГНАЛЫ) ---
  // Используем input() вместо @Input().
  // .required() означает, что значение обязательно (опционально можно убрать).
  lastSeenTimestamp = input<number | null>(null);
  isPresentNow = input<boolean>(false);
  warningThreshold = input<number>(180);
  criticalThreshold = input<number>(300); // get params from backend server
  alertMessage = input<string>('⚠️ Отправлено в Telegram');

  // --- Внутреннее состояние ---
  private secondsAgo = signal(0);
  private timerInterval: any;

  constructor() {
    // Effect сработает при изменении ЛЮБОГО из сигналов, прочитанных внутри него.
    effect(() => {
      // Читаем значения сигналов внутри effect. Это и есть подписка на изменения.
      const timestamp = this.lastSeenTimestamp(); 
      const warning = this.warningThreshold();
      const critical = this.criticalThreshold();
      
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

  // Вычисляемый статус (используем .() для чтения сигналов)
  status = computed(() => {
    if (this.isPresentNow()) return 'present';
    const secs = this.secondsAgo();
    
    if (secs < this.warningThreshold()) return 'present';
    if (secs < this.criticalThreshold()) return 'absent';
    return 'critical';
  });

  // Форматированное время
  displayTime = computed(() => {
    const secs = this.secondsAgo();
    if (this.isPresentNow() || secs < 5) return 'Сейчас';
    
    if (secs < 60) return `${secs} сек. назад`;
    const mins = Math.floor(secs / 60);
    return `${mins} мин. назад`;
  });

  // Иконка
  iconClass = computed(() => {
    switch(this.status()) {
      case 'present': return 'fas fa-user-check';
      case 'absent': return 'fas fa-user-clock';
      case 'critical': return 'fas fa-user-slash';
      default: return 'fas fa-user';
    }
  });

  private updateTimeAgo(timestamp: number): void {
    const now = Math.floor(Date.now() / 1000);
    this.secondsAgo.set(Math.max(0, now - timestamp));
  }
}