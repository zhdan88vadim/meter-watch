import os
from utils.augmentation import AdaptivePreprocess, ExtractLetterWithMargin, OnlyBrighten, RemoveSmallObjects, SquarePadAdaptBackground
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import uuid

def process_dataset(dataset_path, transform, output_suffix="_aug"):
    """
    Applies transformations to each image in the dataset and saves them alongside the originals
    
    Args:
        dataset_path: Path to the root folder of the dataset (with nested class folders)
        transform: Composition of transformations
        output_suffix: Suffix for saved files
    """
    dataset_path = Path(dataset_path)
    
    # Iterate through all files in the dataset (including nested folders)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Check if it's an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = Path(root) / file
                
                try:
                    img = Image.open(img_path)
                    
                    # Apply transformations
                    transformed = transform(img)

                    unique_id = str(uuid.uuid4())[:8]
                    
                    # Save alongside the original
                    new_filename = f"{Path(file).stem}_{unique_id}_{output_suffix}{Path(file).suffix}"
                    output_path = Path("/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val/") / img_path.parent.name / new_filename

                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # output_path = img_path.parent / new_filename
                    
                    # If transformation returned a tensor, convert back to PIL Image
                    if isinstance(transformed, torch.Tensor):
                        # Denormalize and convert to image
                        transformed = transformed.squeeze().cpu()
                        # If values are in range [-1, 1], return to [0, 1]
                        if transformed.min() < 0:
                            transformed = (transformed + 1) / 2
                        transformed = transformed.clamp(0, 1) * 255
                        transformed = transformed.byte().numpy()
                        
                        # Save as PIL Image
                        if len(transformed.shape) == 2:  # Grayscale
                            img_to_save = Image.fromarray(transformed, mode='L')
                        else:  # RGB
                            img_to_save = Image.fromarray(transformed.transpose(1, 2, 0))
                        img_to_save.save(output_path)
                    else:
                        # If transformation returned a PIL Image
                        transformed.save(output_path)
                    
                    print(f"✓ Saved: {output_path}")
                    
                except Exception as e:
                    print(f"✗ Error processing {img_path}: {e}")

if __name__ == "__main__":
    dataset_path = "../../../dataset_train/"
    
    image_size = (28, 28) 
    
    adaptive_preprocess_params = {
        'blur_ksize': 7,           # Reduced from 7 to 3
        'blur_sigma': 5,           # Reduced from 5 to 1
        'adaptive_block_size': 57, # Reduced from 57 to 11 (must be > 1 and odd)
        'adaptive_c': 5,           # Reduced from 5 to 3
        'morph_kernel': 2,         # Reduced from 2 to 1
        'morph_iter': 1            # Kept at 1
    }

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ExtractLetterWithMargin(margin=20, fill_white=None),
        SquarePadAdaptBackground(min_size=128),
        AdaptivePreprocess(apply_prob=1, params=adaptive_preprocess_params),
        transforms.RandomRotation(5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.01, 0.1),
            shear=4
        ),    
        transforms.CenterCrop((90, 90)),
        transforms.Resize((28, 28)),
        RemoveSmallObjects(min_area=5, apply_prob=0.5),
        OnlyBrighten(max_brightness=2.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    process_dataset(dataset_path, transform, output_suffix="_aug")