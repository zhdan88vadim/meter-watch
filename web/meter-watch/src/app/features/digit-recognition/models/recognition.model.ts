export interface DigitResult {
  digit_image: string;      // base64
  heatmap_gradcam: string;  // base64
  heatmap_saliency: string; // base64
  prediction: number;
  confidence: number;
  position: number;
}

export interface RecognitionResponse {
  filename: string;
  full_number: string;
  digits: DigitResult[];
  error?: string;
}

export interface FileListResponse {
  files: string[];
}

export interface SaveDigitRequest {
  digit: number;
  image_base64: string;
  filename: string;
}

export interface DeleteFileRequest {
  filename: string;
  dataset: Dataset;
}

export type Dataset = 'wrong_predictions' | 'validation';