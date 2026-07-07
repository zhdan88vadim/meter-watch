import os
from utils.augmentation import Binarize, ExtractLetterWithMargin, OnlyBrighten, SquarePad, SquarePadAdaptBackground
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import uuid

def process_dataset(dataset_path, transform, output_suffix="_aug"):
    """
    Применяет трансформации к каждому изображению в датасете и сохраняет рядом с оригиналом
    
    Args:
        dataset_path: Путь к корневой папке датасета (с вложенными папками классов)
        transform: Композиция трансформаций
        output_suffix: Суффикс для сохраненных файлов
    """
    dataset_path = Path(dataset_path)
    
    # Проходим по всем файлам в датасете (включая вложенные папки)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Проверяем, что это изображение
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = Path(root) / file
                
                try:
                    # Загружаем изображение
                    img = Image.open(img_path)
                    
                    # Применяем трансформации
                    transformed = transform(img)

                    unique_id = str(uuid.uuid4())[:8]
                    
                    # Сохраняем рядом с оригиналом
                    new_filename = f"{Path(file).stem}_{unique_id}_{output_suffix}{Path(file).suffix}"
                    output_path = Path("/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val/") / img_path.parent.name / new_filename

                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # output_path = img_path.parent / new_filename
                    
                    # Если трансформация вернула тензор, конвертируем обратно в PIL Image
                    if isinstance(transformed, torch.Tensor):
                        # Денормализуем и конвертируем в изображение
                        transformed = transformed.squeeze().cpu()
                        # Если значения в диапазоне [-1, 1], возвращаем в [0, 1]
                        if transformed.min() < 0:
                            transformed = (transformed + 1) / 2
                        transformed = transformed.clamp(0, 1) * 255
                        transformed = transformed.byte().numpy()
                        
                        # Сохраняем как PIL Image
                        if len(transformed.shape) == 2:  # Grayscale
                            img_to_save = Image.fromarray(transformed, mode='L')
                        else:  # RGB
                            img_to_save = Image.fromarray(transformed.transpose(1, 2, 0))
                        img_to_save.save(output_path)
                    else:
                        # Если трансформация вернула PIL Image
                        transformed.save(output_path)
                    
                    print(f"✓ Saved: {output_path}")
                    
                except Exception as e:
                    print(f"✗ Error processing {img_path}: {e}")

# Использование
if __name__ == "__main__":
    # Путь к вашему датасету
    # dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_test/"
    dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_clean/"
    
    # Ваши трансформации
    image_size = (28, 28)  # Или другой размер
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # Binarize(threshold=100, fill_white=False),  # Раскомментируйте если нужен
        ExtractLetterWithMargin(margin=20, fill_white=None),
        SquarePadAdaptBackground(min_size=128),
        # transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,  # Измените на нужные значения
            translate=(0.1, 0.1),
            shear=0
        ),    
        transforms.CenterCrop((90, 90)),
        transforms.Resize((28, 28)),
        OnlyBrighten(max_brightness=2.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    process_dataset(dataset_path, transform, output_suffix="_aug")