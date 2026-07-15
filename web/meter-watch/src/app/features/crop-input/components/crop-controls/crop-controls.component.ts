import {
  Component,
  EventEmitter,
  Input,
  Output,
  OnInit,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormControl } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { SliderModule } from 'primeng/slider';
import { CropParams } from '../../models/crop.models';

@Component({
  selector: 'app-crop-controls',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, ButtonModule, SliderModule],
  template: `
    <div class="control-group">
      <h3>✂️ Параметры обрезки</h3>

      <div class="param-slider">
        <label
          >Обрезка сверху:
          <span class="param-value">{{ cropForm.get('crop_top')?.value }} px</span></label
        >
        <p-slider
          [formControl]="getControl('crop_top')"
          [min]="0"
          [max]="maxTop"
          [step]="5"
          (onChange)="onParamChange('crop_top', $event.value)"
        >
        </p-slider>
      </div>

      <div class="param-slider">
        <label
          >Обрезка снизу:
          <span class="param-value">{{ cropForm.get('crop_bottom')?.value }} px</span></label
        >
        <p-slider
          [formControl]="getControl('crop_bottom')"
          [min]="0"
          [max]="maxBottom"
          [step]="5"
          (onChange)="onParamChange('crop_bottom', $event.value)"
        >
        </p-slider>
      </div>

      <div class="param-slider">
        <label
          >Обрезка слева:
          <span class="param-value">{{ cropForm.get('crop_left')?.value }} px</span></label
        >
        <p-slider
          [formControl]="getControl('crop_left')"
          [min]="0"
          [max]="maxLeft"
          [step]="5"
          (onChange)="onParamChange('crop_left', $event.value)"
        >
        </p-slider>
      </div>

      <div class="param-slider">
        <label
          >Обрезка справа:
          <span class="param-value">{{ cropForm.get('crop_right')?.value }} px</span></label
        >
        <p-slider
          [formControl]="getControl('crop_right')"
          [min]="0"
          [max]="maxRight"
          [step]="5"
          (onChange)="onParamChange('crop_right', $event.value)"
        >
        </p-slider>
      </div>

      <div class="current-values">
        <strong>Текущие значения:</strong><br />
        Сверху: {{ cropForm.get('crop_top')?.value }}px | Снизу:
        {{ cropForm.get('crop_bottom')?.value }}px | Слева: {{ cropForm.get('crop_left')?.value }}px
        | Справа: {{ cropForm.get('crop_right')?.value }}px
      </div>
    </div>
  `,
  styles: [
    `
      .control-group {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
      }
      .control-group h3 {
        color: #333;
        margin-bottom: 15px;
        font-size: 1.2em;
      }
      .param-slider {
        margin-bottom: 20px;
      }
      .param-slider label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: #555;
      }
      .param-value {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.9em;
        margin-left: 10px;
      }
      .current-values {
        background: #e9ecef;
        padding: 10px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 0.9em;
      }
    `,
  ],
})
export class CropControlsComponent implements OnInit, OnChanges {
  @Input() params!: CropParams;
  @Input() maxTop = 300;
  @Input() maxBottom = 300;
  @Input() maxLeft = 300;
  @Input() maxRight = 300;
  @Output() paramChange = new EventEmitter<{ param: keyof CropParams; value: number }>();

  cropForm: FormGroup;

  constructor(private fb: FormBuilder) {
    this.cropForm = this.fb.group({
      crop_top: [0],
      crop_bottom: [0],
      crop_left: [0],
      crop_right: [0],
    });
  }

  ngOnInit(): void {
    this.updateFormFromParams();
  }

  ngOnChanges(changes: SimpleChanges): void {
    // Sync form when @Input params changes from parent
    if (changes['params'] && !changes['params'].firstChange) {
      this.updateFormFromParams();
    }
  }

  private updateFormFromParams(): void {
    // emitEvent: false prevents triggering valueChanges subscription
    this.cropForm.patchValue(
      {
        crop_top: this.params.crop_top,
        crop_bottom: this.params.crop_bottom,
        crop_left: this.params.crop_left,
        crop_right: this.params.crop_right,
      },
      { emitEvent: false },
    );
  }

  onParamChange(param: keyof CropParams, value: number = 0): void {
    this.paramChange.emit({ param, value });
  }

  getControl(name: keyof CropParams): FormControl<number> {
    return this.cropForm.get(name) as FormControl<number>;
  }
}
