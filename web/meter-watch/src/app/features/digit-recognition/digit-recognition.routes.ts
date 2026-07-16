import { Routes } from '@angular/router';
import { DigitRecognitionComponent } from './components/digit-recognition/digit-recognition';

export const DIGIT_RECOGNITION_ROUTES: Routes = [
  {
    path: '',
    component: DigitRecognitionComponent,
    title: 'Digit Recognition'
  }
];