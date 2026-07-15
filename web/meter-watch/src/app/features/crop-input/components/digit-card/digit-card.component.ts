import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Digit } from '../../models/crop.models';

@Component({
  selector: 'app-digit-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="digit-card">
      <div class="prediction">{{digit.prediction}}</div>
      <img [src]="'data:image/png;base64,' + digit.digit_image" [alt]="'Digit ' + digit.position">
      <div class="confidence">Уверенность: {{digit.confidence * 100 | number:'1.0-1'}}%</div>
      <div class="info-text">Позиция: {{digit.position}}</div>
    </div>
  `,
  styles: [`
    .digit-card {
      background: white;
      border-radius: 8px;
      padding: 10px;
      text-align: center;
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s;
    }
    .digit-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .digit-card img {
      width: 80px;
      height: 80px;
      margin: 10px auto;
    }
    .prediction {
      font-size: 2em;
      font-weight: bold;
      color: #667eea;
    }
    .confidence {
      font-size: 0.9em;
      color: #666;
      margin-top: 5px;
    }
    .info-text {
      font-size: 0.8em;
      color: #999;
      margin-top: 5px;
    }
  `]
})
export class DigitCardComponent implements OnInit {
  @Input() digit!: Digit;

  ngOnInit(): void {
    console.log(this.digit);
    
  }
}