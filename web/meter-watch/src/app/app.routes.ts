import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'recognition', pathMatch: 'full' },
  {
    path: 'recognition',
    loadChildren: () =>
      import('./features/digit-recognition/digit-recognition.routes').then(
        (m) => m.DIGIT_RECOGNITION_ROUTES,
      ),
  },
];
