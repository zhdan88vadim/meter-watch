import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { finalize, firstValueFrom, forkJoin } from 'rxjs';

import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { ToastModule } from 'primeng/toast';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { ConfirmationService, MessageService } from 'primeng/api';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { PaginatorModule, PaginatorState } from 'primeng/paginator';

import { ApiService } from '../../../../core/services/api.service';
import { DatasetSelectorComponent } from '../../../../shared/components/dataset-selector/dataset-selector.component';
import {
  Dataset,
  RecognitionResponse,
  SaveDigitRequest,
} from '../../models/recognition.model';
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
    PaginatorModule,
    DatasetSelectorComponent,
    DigitCardComponent,
    DigitDisplayComponent,
  ],
  providers: [MessageService, ConfirmationService],
  templateUrl: './digit-recognition.html',
  styleUrl: './digit-recognition.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DigitRecognitionComponent {
  private readonly api = inject(ApiService);
  private readonly messageService = inject(MessageService);
  private readonly confirmationService = inject(ConfirmationService);

  // --- state ---
  readonly dataset = signal<Dataset>('wrong_predictions');
  readonly files = signal<readonly string[]>([]);
  readonly loading = signal(false);
  readonly processing = signal(false);
  readonly currentFile = signal<string | null>(null);
  readonly recognitionResult = signal<RecognitionResponse | null>(null);
  readonly selectedDigits = signal<ReadonlyMap<number, number>>(new Map());

  // --- pagination ---
  readonly currentPage = signal(1);
  readonly pageSize = signal(10);
  readonly totalRecords = computed(() => this.files().length);

  readonly paginatedFiles = computed(() => {
    const start = (this.currentPage() - 1) * this.pageSize();
    return this.files().slice(start, start + this.pageSize());
  });

  readonly firstRecordIndex = computed(
    () => (this.currentPage() - 1) * this.pageSize(),
  );

  // --- derived ---
  readonly hasSelections = computed(() => this.selectedDigits().size > 0);

  constructor() {
    this.loadFiles();
  }

  // --- data loading ---
  loadFiles(): void {
    this.loading.set(true);
    this.api
      .getFiles(this.dataset())
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: (files) => {
          this.files.set(files);
          this.currentPage.set(1);
          if (files.length) this.recognize(files[0]);
        },
        error: (err: Error) =>
          this.showError('Failed to load files', err.message),
      });
  }

  onDatasetChange(newDataset: Dataset): void {
    this.dataset.set(newDataset);
    this.resetSelection();
    this.loadFiles();
  }

  onPageChange(event: PaginatorState): void {
    const rows = event.rows ?? this.pageSize();
    this.pageSize.set(rows);
    this.currentPage.set(((event.first ?? 0) / rows) + 1);
  }

  // --- recognition ---
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
          this.recognitionResult.set(result);
        },
        error: (err: Error) => this.showError('Recognition failed', err.message),
      });
  }

  // --- deletion ---
  requestDelete(filename: string): void {
    this.confirmationService.confirm({
      message: `Delete file "${filename}"?`,
      header: 'Confirm Delete',
      icon: 'pi pi-exclamation-triangle',
      acceptButtonStyleClass: 'p-button-danger',
      accept: () => this.deleteFile(filename),
    });
  }

  private deleteFile(filename: string): void {
    this.api.deleteFile({ filename, dataset: this.dataset() }).subscribe({
      next: (res) => {
        if (!res.success) {
          this.showError('Delete failed', 'Server returned success=false');
          return;
        }
        this.showSuccess(`Deleted: ${filename}`);
        this.loadFiles();
        if (this.currentFile() === filename) this.resetSelection();
      },
      error: (err: Error) => this.showError('Delete failed', err.message),
    });
  }

  // --- selections ---
  onDigitSelected(event: { position: number; digit: number }): void {
    const map = new Map(this.selectedDigits());
    map.set(event.position, event.digit);
    this.selectedDigits.set(map);
  }

  onDigitDeselected(position: number): void {
    const map = new Map(this.selectedDigits());
    map.delete(position);
    this.selectedDigits.set(map);
  }

  async submitCorrections(): Promise<void> {
    const file = this.currentFile();
    if (!file || !this.hasSelections()) {
      this.messageService.add({
        severity: 'warn',
        summary: 'No selections',
        detail: 'Please select at least one digit correction',
      });
      return;
    }

    const baseName = file.replace('.png', '');
    const digits = this.recognitionResult()?.digits ?? [];
    const requests: SaveDigitRequest[] = [];

    this.selectedDigits().forEach((digit, position) => {
      const d = digits[position];
      if (d?.digit_image) {
        requests.push({
          digit,
          image_base64: d.digit_image,
          filename: `${baseName}_pos${position + 1}_digit${digit}.png`,
        });
      }
    });

    if (!requests.length) {
      this.showError('Save failed', 'Nothing to save');
      return;
    }

    this.processing.set(true);
    try {
      const responses = await firstValueFrom(forkJoin(requests.map((r) => this.api.saveDigit(r))));
      if (!responses.every((r) => r?.success)) {
        throw new Error('Some saves failed');
      }
      const info = [...this.selectedDigits()]
        .map(([p, d]) => `Pos ${p + 1} → ${d}`)
        .join(', ');
      this.showSuccess(`Saved ${requests.length} digit(s): ${info}`);
      this.selectedDigits.set(new Map());
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      this.showError('Save failed', msg);
    } finally {
      this.processing.set(false);
    }
  }

  // --- helpers ---
  private resetSelection(): void {
    this.currentFile.set(null);
    this.recognitionResult.set(null);
    this.selectedDigits.set(new Map());
  }

  private showSuccess(detail: string): void {
    this.messageService.add({ severity: 'success', summary: 'Success', detail, life: 3000 });
  }

  private showError(summary: string, detail: string): void {
    this.messageService.add({ severity: 'error', summary, detail, life: 5000 });
  }
}
