import {
  Component,
  Input,
  ViewChild,
  ElementRef,
  AfterViewInit,
  EventEmitter,
  Output,
  OnChanges,
  SimpleChanges,
  OnInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { CropParams } from '../../models/crop.models';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-crop-preview',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (imageBase64) {
      <div class="image-container">
        <h3>📸 Исходное изображение с линиями обрезки</h3>
        <div class="original-image-wrapper">
          <img
            #originalImage
            [src]="'data:image/png;base64,' + imageBase64"
            alt="Исходное изображение"
          />
          <canvas
            #cropCanvas
            [attr.width]="imageWidth"
            [attr.height]="imageHeight"
            style="position: absolute; top: 0; left: 0; cursor: crosshair;"
          ></canvas>
        </div>
      </div>
    }
  `,
  styles: [
    `
      .image-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
      }
      .image-container h3 {
        margin-bottom: 15px;
        color: #333;
      }
      .original-image-wrapper {
        position: relative;
        display: inline-block;
        max-width: 100%;
      }
      .original-image-wrapper img {
        display: block;
        max-width: 100%;
        height: auto;
      }
      .original-image-wrapper canvas {
        position: absolute;
        top: 0;
        left: 0;
        cursor: crosshair;
        width: auto;
        max-width: 100%;
        height: auto;
      }
    `,
  ],
})
export class CropPreviewComponent implements AfterViewInit, OnChanges, OnInit {
  @ViewChild('originalImage') originalImageRef!: ElementRef<HTMLImageElement>;
  @ViewChild('cropCanvas') cropCanvasRef!: ElementRef<HTMLCanvasElement>;

  @Input() imageBase64: string | null = null;
  @Input() params!: CropParams;
  @Input() imageWidth = 0;
  @Input() imageHeight = 0;
  @Output() paramsChange = new EventEmitter<CropParams>();

  @Output() paramChange = new EventEmitter<{ param: keyof CropParams; value: number }>();

  isDrawing = false;
  startX = 0;
  startY = 0;
  drawRect = { x: 0, y: 0, w: 0, h: 0 };

  constructor(private messageService: MessageService) {}

  ngAfterViewInit(): void {
    this.setupCanvasEvents();
  }

  ngOnInit(): void {
    setTimeout(() => {
      this.drawCropRectangle();
    }, 0);
  }

  updateParam(param: keyof CropParams, value: number = 0): void {
    this.paramChange.emit({ param, value });
  }

  private setupCanvasEvents(): void {
    const canvas = this.cropCanvasRef.nativeElement;
    if (!canvas) return;

    // Удаляем старые слушатели, если есть
    canvas.removeEventListener('mousedown', this.onMouseDown.bind(this));
    canvas.removeEventListener('mousemove', this.onMouseMove.bind(this));
    canvas.removeEventListener('mouseup', this.onMouseUp.bind(this));
    canvas.removeEventListener('mouseleave', this.onMouseUp.bind(this));

    // Добавляем новые
    canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
    canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
    canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
  }

  ngOnChanges(changes: SimpleChanges): void {
    console.log('ngOnChanges');

    if (this.imageWidth > 0) {
      this.drawRect = {
        x: this.params.crop_left,
        y: this.params.crop_top,
        w: this.imageWidth - this.params.crop_left - this.params.crop_right,
        h: this.imageHeight - this.params.crop_top - this.params.crop_bottom,
      };
      this.drawCropRectangle();
    }
  }

  // Рисование прямоугольника
  private drawCropRectangle(): void {
    const canvas = this.cropCanvasRef?.nativeElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = this.imageWidth;
    canvas.height = this.imageHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = this.drawRect;
    const params = this.params;

    if (rect.w !== 0 && rect.h !== 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.clearRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 3;
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    } else {
      const cropX = params.crop_left;
      const cropY = params.crop_top;
      const cropW = this.imageWidth - params.crop_left - params.crop_right;
      const cropH = this.imageHeight - params.crop_top - params.crop_bottom;

      if (cropW > 0 && cropH > 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.clearRect(cropX, cropY, cropW, cropH);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 3;
        ctx.strokeRect(cropX, cropY, cropW, cropH);
      }
    }
  }

  // События мыши
  private onMouseDown(e: MouseEvent): void {
    const canvas = this.cropCanvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const scaleX = this.imageWidth / canvas.clientWidth;
    const scaleY = this.imageHeight / canvas.clientHeight;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    this.isDrawing = true;
    this.startX = mouseX;
    this.startY = mouseY;
    this.drawRect = { x: this.startX, y: this.startY, w: 0, h: 0 };
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.isDrawing) return;

    const canvas = this.cropCanvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const scaleX = this.imageWidth / canvas.clientWidth;
    const scaleY = this.imageHeight / canvas.clientHeight;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const w = mouseX - this.startX;
    const h = mouseY - this.startY;

    this.drawRect = { x: this.startX, y: this.startY, w, h };
    this.drawCropRectangle();
  }

  private onMouseUp(e: MouseEvent): void {
    if (this.isDrawing) {
      let rect = this.drawRect;

      // Нормализуем координаты
      if (rect.w < 0) {
        rect = { x: rect.x + rect.w, y: rect.y, w: -rect.w, h: rect.h };
      }
      if (rect.h < 0) {
        rect = { x: rect.x, y: rect.y + rect.h, w: rect.w, h: -rect.h };
      }

      if (rect.w > 5 && rect.h > 5) {
        // Обновляем параметры обрезки
        const newParams: CropParams = {
          crop_top: Math.max(0, Math.min(this.imageHeight, Math.round(rect.y))),
          crop_left: Math.max(0, Math.min(this.imageWidth, Math.round(rect.x))),
          crop_bottom: Math.max(
            0,
            Math.min(this.imageHeight, Math.round(this.imageHeight - (rect.y + rect.h))),
          ),
          crop_right: Math.max(
            0,
            Math.min(this.imageWidth, Math.round(this.imageWidth - (rect.x + rect.w))),
          ),
        };
        this.params = newParams;
        this.drawRect = {
          x: this.params.crop_left,
          y: this.params.crop_top,
          w: this.imageWidth - this.params.crop_left - this.params.crop_right,
          h:
            this.imageHeight - this.params.crop_top - this.params.crop_bottom,
        };

        this.paramsChange.emit(this.params);

        this.drawCropRectangle();
      }
    }

    this.isDrawing = false;
  }
}
