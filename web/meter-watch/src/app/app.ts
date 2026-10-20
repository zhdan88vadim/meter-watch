import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { MiniHeaderComponent } from './shared/components/mini-header/mini-header.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, MiniHeaderComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {}
