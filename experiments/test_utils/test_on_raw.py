import os
import uuid
import shutil
from pathlib import Path
import cv2
from typing import Dict, List, Any
import json
import numpy as np

from models.pytorch_model import load_pytorch_model
from services.recognition import recognize_image

def save_rename_history(history: List[Dict], log_file: str, log_format: str = "json"):
    """Saves rename history to a file"""
    
    if log_format in ["json", "both"]:
        json_file = log_file if log_file.endswith('.json') else f"{log_file}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON log saved: {json_file}")

def denormalize_image(normalized):
    """
    Converts a normalized image back to a regular format for display
    
    Args:
        normalized: normalized image (values from -1 to 1)
    
    Returns:
        image in uint8 format (0-255)
    """
    # Denormalization: (x * 0.5) + 0.5
    denormalized = (normalized * 0.5) + 0.5
    
    # Clip values outside range [0, 1]
    denormalized = np.clip(denormalized, 0, 1)
    
    # Convert to uint8 (0-255)
    denormalized = (denormalized * 255).astype(np.uint8)
    
    return denormalized

def process_and_rename_images(input_dir, output_dir, recognize_image_func):
    """
    Processes all images from a directory, recognizes them and renames
    
    Args:
        input_dir: path to directory with source images
        output_dir: path to directory for saving processed images
        recognize_image_func: recognition function returning (result, min_conf)
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in directory {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    processed_count = 0
    failed_count = 0
    rename_history = []
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        try:
            
            image = cv2.imread(img_path)

            result, min_conf = recognize_image(image)


            for digit in result['digits']:

                # if digit['confidence'] > 0.5:
                unique_id = str(uuid.uuid4())[:8]
                dir = f"../../dataset_val_test_raw/{digit['prediction']}/"
                filepath = os.path.join(dir, f"{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                filepath_original = os.path.join(dir, f"orgnl__{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                filepath_model = os.path.join(dir, f"model__{digit['prediction']}__{digit['confidence']:.2f}_{unique_id}.jpg")
                print(filepath)
                Path(dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filepath, digit['raw_image'])
                cv2.imwrite(filepath_original, digit['source_image'])
                # cv2.imwrite(filepath_model, denormalize_image(digit['prepared_model_image']))
        
            # Get recognized number
            new_digits = list(result['full_number'])
            recognized_number = result['full_number']
            
            # If number not recognized or empty
            if not recognized_number:
                print(f"⚠️  Failed to recognize number on {img_file}, skipping")
                failed_count += 1
                continue
            
            # Generate unique ID (first 8 characters for brevity)
            unique_id = str(uuid.uuid4())[:8]
            
            # Form new filename
            original_extension = Path(img_file).suffix
            new_filename = f"{recognized_number}_{unique_id}{original_extension}"
            new_path = os.path.join(output_dir, new_filename)
            
            # Copy and rename file
            shutil.copy2(img_path, new_path)
            
            print(f"✓ {img_file} -> {new_filename} (confidence: {min_conf:.2f})")
            processed_count += 1

            # Save rename information
            history_entry = {
                'old_name': img_file,
                'new_name': new_filename,
                'recognized_number': recognized_number,
                'confidence': min_conf,
            }
            rename_history.append(history_entry)
            
        except Exception as e:
            print(f"✗ Error processing {img_file}: {str(e)}")
            failed_count += 1
    

    save_rename_history(rename_history, "rename_history.json", "json")

    # Print statistics
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Errors/skipped: {failed_count}")
    print(f"📁 Results saved to: {output_dir}")
    print("="*50)


# Example usage with pseudo-recognition function
# (replace with your actual function)

def example_recognize_image(img_path):
    """
    Example recognition function.
    Replace with your actual implementation.
    """
    # Your actual recognition logic here
    # Should return (result, min_conf)
    
    # Example for testing:
    class Result:
        pass
    
    result = Result()
    # Assume result contains 'full_number' field
    result['full_number'] = "12345"  # example recognized number
    
    min_conf = 0.95
    
    return result, min_conf


if __name__ == "__main__":
    load_pytorch_model()
    
    # INPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/hard"
    INPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/raw_images"
    OUTPUT_DIRECTORY = "/media/vadim/1TB_SSD/my_github/meter-watch/hard_images_output"
    
    path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_test_raw/"

    # Get only directories
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dir_name in dirs:
        dir_path = os.path.join(path, dir_name)
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_name}")

    process_and_rename_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, example_recognize_image)