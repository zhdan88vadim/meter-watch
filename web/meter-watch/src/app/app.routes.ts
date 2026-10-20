import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'recognition' },
  {
    path: 'recognition',
    title: 'MeterWatch — Recognition',
    loadChildren: () =>
      import('./features/digit-recognition/digit-recognition.routes').then(
        (m) => m.DIGIT_RECOGNITION_ROUTES,
      ),
  },
  { path: '**', redirectTo: 'recognition' },
];
