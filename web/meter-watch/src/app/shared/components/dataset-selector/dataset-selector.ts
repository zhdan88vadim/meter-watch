import { Component, EventEmitter, Output, input } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { Dataset } from '../../../features/digit-recognition/models/recognition.model';

@Component({
  selector: 'app-dataset-selector',
  standalone: true,
  imports: [ButtonModule],
  template: `
    <div class="flex gap-2 flex-wrap">
      @for (ds of datasets; track ds) {
        <p-button
          [label]="getLabel(ds)"
          [severity]="selectedDataset() === ds ? 'success' : 'secondary'"
          [outlined]="selectedDataset() !== ds"
          (onClick)="onSelect(ds)"
        />
      }
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
        margin-bottom: 1rem;
      }
    `,
  ],
})
export class DatasetSelectorComponent {
  readonly selectedDataset = input.required<Dataset>();
  @Output() datasetChange = new EventEmitter<Dataset>();

  readonly datasets: Dataset[] = ['wrong_predictions', 'validation'];

  getLabel(ds: Dataset): string {
    const labels: Record<Dataset, string> = {
      wrong_predictions: '❌ Wrong Predictions',
      validation: '✅ Validation Dataset',
    };
    return labels[ds];
  }

  onSelect(dataset: Dataset) {
    this.datasetChange.emit(dataset);
  }
}
