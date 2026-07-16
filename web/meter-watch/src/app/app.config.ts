// src/app/app.config.ts
import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { routes } from './app.routes';

// 1. Import PrimeNG Config
import { providePrimeNG } from 'primeng/config';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

// 2. Import Animations (Required for Dialogs, Dropdowns, etc.)
import { provideAnimations } from '@angular/platform-browser/animations';
import Material from '@primeuix/themes/material';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),

    // 3. Add PrimeNG Configuration
    providePrimeNG({
      ripple: true, // Enables ripple effect on buttons
      inputStyle: 'filled', // 'filled' or 'outlined',
      theme: {
        preset: Material,
        options: {
          // Отключаем тёмный режим полностью (опционально)
          darkModeSelector: false, // или 'none'
        },
      },
    }),

    // 4. Add Animations
    provideAnimations(),
  ],
};
