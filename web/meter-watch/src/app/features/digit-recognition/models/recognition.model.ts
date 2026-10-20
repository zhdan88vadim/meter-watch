export interface DigitResult {
  readonly digit_image: string;      // base64
  readonly heatmap_gradcam: string;  // base64
  readonly heatmap_saliency: string; // base64
  readonly prediction: number;
  readonly confidence: number;
  readonly position: number;
}

export interface RecognitionResponse {
  readonly filename: string;
  readonly full_number: string;
  readonly digits: readonly DigitResult[];
  readonly error?: string;
}

export interface FileListResponse {
  readonly files: readonly string[];
}

export interface SaveDigitRequest {
  readonly digit: number;
  readonly image_base64: string;
  readonly filename: string;
}

export interface DeleteFileRequest {
  readonly filename: string;
  readonly dataset: Dataset;
}

export interface SaveResult {
  readonly success: boolean;
}

export type Dataset = 'wrong_predictions' | 'validation';
