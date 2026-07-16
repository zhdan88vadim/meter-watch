import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs/operators';

// PrimeNG modules
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { ToastModule } from 'primeng/toast';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { ConfirmationService, MessageService } from 'primeng/api';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { BadgeModule } from 'primeng/badge';
import { PaginatorModule } from 'primeng/paginator';
import { PaginatorState } from 'primeng/paginator';

// Local components & services
import { ApiService } from '../../../../core/services/api.service';
import { DatasetSelectorComponent } from '../../../../shared/components/dataset-selector/dataset-selector';
import { Dataset, RecognitionResponse, SaveDigitRequest } from '../../models/recognition.model';
import { DigitCardComponent } from '../digit-card/digit-card';
import { DigitDisplayComponent } from '../../../digit-display/components/digit-display/digit-display.component';

@Component({
  selector: 'app-digit-recognition',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ButtonModule,
    CardModule,
    ToastModule,
    ConfirmDialogModule,
    ProgressSpinnerModule,
    BadgeModule,
    PaginatorModule,
    DatasetSelectorComponent,
    DigitCardComponent,
    DigitDisplayComponent,
  ],
  providers: [MessageService, ConfirmationService],
  templateUrl: './digit-recognition.html',
  styleUrl: './digit-recognition.scss',
})
export class DigitRecognitionComponent {
  private readonly api = inject(ApiService);
  private readonly messageService = inject(MessageService);
  private readonly confirmationService = inject(ConfirmationService);

  // State signals (Angular 18)
  readonly dataset = signal<Dataset>('wrong_predictions');
  readonly files = signal<string[]>([]);
  readonly loading = signal(false);
  readonly processing = signal(false);
  readonly currentFile = signal<string | null>(null);
  readonly recognitionResult = signal<RecognitionResponse | null>(null);
  readonly selectedDigits = signal<Map<number, number>>(new Map()); // position -> digit

  // Pagination signals
  readonly currentPage = signal<number>(1);
  readonly pageSize = signal<number>(10);
  readonly totalRecords = computed(() => this.files().length);

  // Computed paginated files
  readonly paginatedFiles = computed(() => {
    const startIndex = (this.currentPage() - 1) * this.pageSize();
    const endIndex = startIndex + this.pageSize();
    return this.files().slice(startIndex, endIndex);
  });

  // Computed
  readonly hasSelections = computed(() => this.selectedDigits().size > 0);
  readonly totalPages = computed(() => Math.ceil(this.totalRecords() / this.pageSize()));

  constructor() {
    this.loadFiles();
  }

  /** Загрузить список файлов для текущего датасета */
  loadFiles(): void {
    this.loading.set(true);
    this.api
      .getFiles(this.dataset())
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: (files) => {
          this.files.set(files);
          // Reset to first page when loading new files
          this.currentPage.set(1);
          if (files.length) {
            this.recognize(files[0]);
          }
        },
        error: (err) => this.showError('Failed to load files', err.message),
      });
  }

  /** Сменить датасет */
  onDatasetChange(newDataset: Dataset): void {
    this.dataset.set(newDataset);
    this.resetSelection();
    this.loadFiles();
  }

  /** Handle page change */
  onPageChange(event: PaginatorState): void {
    const newPage = (event.first ?? 0) / (event.rows ?? this.pageSize()) + 1;
    this.currentPage.set(newPage);
    this.pageSize.set(event.rows ?? this.pageSize());
  }

  /** Распознать файл */
  recognize(filename: string): void {
    this.currentFile.set(filename);
    this.processing.set(true);
    this.recognitionResult.set(null);
    this.selectedDigits.set(new Map());

    this.api
      .recognize(filename, this.dataset())
      .pipe(finalize(() => this.processing.set(false)))
      .subscribe({
        next: (result) => {
          if (result.error) {
            this.showError('Recognition error', result.error);
            return;
          }

          result.digits = result.digits; // .filter(digit => digit.confidence < 0.9);

          this.recognitionResult.set(result);
        },
        error: (err) => this.showError('Recognition failed', err.message),
      });
  }

  /** Удалить файл с подтверждением */
  requestDelete(filename: string): void {
    this.confirmationService.confirm({
      message: `Delete file "${filename}"?`,
      header: 'Confirm Delete',
      icon: 'pi pi-exclamation-triangle',
      acceptButtonStyleClass: 'p-button-danger',
      accept: () => this.deleteFile(filename),
      reject: () => {},
    });
  }

  /** Удалить файл (после подтверждения) */
  private deleteFile(filename: string): void {
    this.api.deleteFile({ filename, dataset: this.dataset() }).subscribe({
      next: (res) => {
        if (res.success) {
          this.showSuccess(`Deleted: ${filename}`);
          this.loadFiles();
          // Очистить результат если удалили текущий файл
          if (this.currentFile() === filename) {
            this.resetSelection();
          }
        }
      },
      error: (err) => this.showError('Delete failed', err.message),
    });
  }

  /** Обработать выбор цифры в карточке */
  onDigitSelected(event: { position: number; digit: number }): void {
    const map = new Map(this.selectedDigits());
    map.set(event.position, event.digit);
    this.selectedDigits.set(map);
  }

  /** Обработать отмену выбора цифры */
  onDigitDeselected(position: number): void {
    const map = new Map(this.selectedDigits());
    map.delete(position);
    this.selectedDigits.set(map);
  }

  /** Сохранить исправления */
  submitCorrections(): void {
    if (!this.currentFile() || !this.hasSelections()) {
      this.messageService.add({
        severity: 'warn',
        summary: 'No selections',
        detail: 'Please select at least one digit correction',
      });
      return;
    }

    const baseFilename = this.currentFile()!.replace('.png', '');
    const digits = this.recognitionResult()?.digits || [];
    const saveRequests: SaveDigitRequest[] = [];

    this.selectedDigits().forEach((digit, position) => {
      const digitData = digits[position];
      if (digitData?.digit_image) {
        saveRequests.push({
          digit,
          image_base64: digitData.digit_image,
          filename: `${baseFilename}_pos${position + 1}_digit${digit}.png`,
        });
      }
    });

    this.processing.set(true);

    // Выполнить все запросы параллельно
    Promise.all(saveRequests.map((req) => this.api.saveDigit(req).toPromise()))
      .then((results) => {
        const allSuccess = results.every((r) => r?.success);

        if (allSuccess) {
          const savedInfo = Array.from(this.selectedDigits())
            .map(([pos, dig]) => `Pos ${pos + 1} → ${dig}`)
            .join(', ');

          this.showSuccess(`Saved ${saveRequests.length} digit(s): ${savedInfo}`);
          this.selectedDigits.set(new Map()); // Очистить выбор
        } else {
          throw new Error('Some saves failed');
        }
      })
      .catch((err) => {
        this.showError('Save failed', err.message);
      })
      .finally(() => {
        this.processing.set(false);
      });
  }

  /** Сбросить выбор */
  private resetSelection(): void {
    this.currentFile.set(null);
    this.recognitionResult.set(null);
    this.selectedDigits.set(new Map());
  }

  /** Показать уведомление об успехе */
  private showSuccess(detail: string): void {
    this.messageService.add({
      severity: 'success',
      summary: 'Success',
      detail,
      life: 3000,
    });
  }

  /** Показать уведомление об ошибке */
  private showError(summary: string, detail: string): void {
    this.messageService.add({
      severity: 'error',
      summary,
      detail,
      life: 5000,
    });
  }
}
