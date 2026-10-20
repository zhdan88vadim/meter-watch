import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-mini-header',
  standalone: true,
  templateUrl: './mini-header.component.html',
  styleUrl: './mini-header.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MiniHeaderComponent {}
