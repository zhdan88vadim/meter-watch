import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, RouterOutlet } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { PasswordModule } from 'primeng/password';
import { MiniHeaderComponent } from './shared/components/mini-header/mini-header.component';
import { DigitDisplayComponent } from './features/digit-display/components/digit-display/digit-display.component';

@Component({
  selector: 'app-root',
  imports: [
    RouterOutlet,
    ButtonModule,
    PasswordModule,
    FormsModule,
    RouterModule,
    MiniHeaderComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected readonly title = signal('meter-watch');
}
