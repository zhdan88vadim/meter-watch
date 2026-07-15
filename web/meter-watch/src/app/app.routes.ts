import { Routes } from '@angular/router';

// app.routes.ts
export const routes: Routes = [
  { path: '', redirectTo: 'recognition', pathMatch: 'full' },
  {
    path: 'crop-editor',
    loadComponent: () =>
      import('./features/crop-input/components/crop-editor/crop-editor.component').then((m) => m.CropEditorComponent),
  },
  {
    path: 'recognition',
    loadChildren: () =>
      import('./features/digit-recognition/digit-recognition.routes').then(
        (m) => m.DIGIT_RECOGNITION_ROUTES,
      ),
  },
  //   {
  //     path: 'users',
  //     loadChildren: () => import('./features/users/users.routes').then((m) => m.USER_ROUTES),
  //   },
];
