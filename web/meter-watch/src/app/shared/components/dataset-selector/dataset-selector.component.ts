import { ChangeDetectionStrategy, Component, EventEmitter, Output, input } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { Dataset } from '../../../features/digit-recognition/models/recognition.model';

@Component({
  selector: 'app-dataset-selector',
  standalone: true,
  imports: [ButtonModule],
  templateUrl: './dataset-selector.component.html',
  styleUrl: './dataset-selector.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DatasetSelectorComponent {
  readonly selectedDataset = input.required<Dataset>();
  @Output() readonly datasetChange = new EventEmitter<Dataset>();

  readonly datasets: readonly Dataset[] = ['wrong_predictions', 'validation'] as const;

  readonly labels: Readonly<Record<Dataset, string>> = {
    wrong_predictions: '❌ Wrong Predictions',
    validation: '✅ Validation Dataset',
  };
}
