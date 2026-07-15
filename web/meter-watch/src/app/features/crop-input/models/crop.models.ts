// src/app/models/crop.models.ts
export interface CropParams {
  crop_top: number;
  crop_bottom: number;
  crop_left: number;
  crop_right: number;
}

export interface CropResult {
  original_image: string;
  cropped_image: string;
  warning: string;
  threshold_image: string;
  digits: Digit[];
  full_number: string;
  params_used: CropParams;
  original_size: ImageSize;
  cropped_size: ImageSize;
}

export interface Digit {
  position: number;
  prediction: number;
  confidence: number;
  digit_image: string;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface ApiResponse<T> {
  status: string;
  result?: T;
  error?: string;
  config?: CropParams;
  message?: string;
  changes?: any;
}

export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}