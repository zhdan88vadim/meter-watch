import os
import random
from pathlib import Path
import shutil

def balance_dataset(dataset_path, target_count=None, keep_random=True):
    """
    Балансирует датасет, удаляя лишние изображения из папок.
    
    Args:
        dataset_path: Путь к корневой папке датасета
        target_count: Желаемое количество изображений в каждой папке.
                     Если None, то используется минимальное количество среди всех папок
        keep_random: Если True, оставляет случайные файлы.
                    Если False, оставляет первые N файлов
    """
    dataset_path = Path(dataset_path)
    
    # Собираем информацию по всем папкам
    folder_stats = {}
    for folder in dataset_path.iterdir():
        if not folder.is_dir():
            continue
            
        # Получаем все изображения в папке (исключая подпапки)
        images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in 
                 ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        
        folder_stats[folder.name] = {
            'path': folder,
            'count': len(images),
            'images': images
        }
        print(f"📁 {folder.name}: {len(images)} images")
    
    # Определяем целевое количество
    if target_count is None:
        target_count = min(stats['count'] for stats in folder_stats.values())
    
    print(f"\n🎯 Target count per folder: {target_count}")
    print("=" * 50)
    
    # Удаляем лишние файлы
    total_deleted = 0
    for folder_name, stats in folder_stats.items():
        current_count = stats['count']
        images = stats['images']
        
        if current_count > target_count:
            # Выбираем, какие файлы удалить
            if keep_random:
                # Случайно выбираем, какие сохранить
                to_keep = random.sample(images, target_count)
                to_delete = [img for img in images if img not in to_keep]
            else:
                # Удаляем последние (по алфавиту) файлы
                images_sorted = sorted(images)
                to_keep = images_sorted[:target_count]
                to_delete = images_sorted[target_count:]
            
            # Удаляем файлы
            for img_path in to_delete:
                try:
                    img_path.unlink()  # Удаляем файл
                    total_deleted += 1
                except Exception as e:
                    print(f"❌ Error deleting {img_path}: {e}")
            
            print(f"📁 {folder_name}: {current_count} → {target_count} (deleted {len(to_delete)})")
        elif current_count < target_count:
            print(f"⚠️  {folder_name}: {current_count} (needs {target_count - current_count} more)")
        else:
            print(f"✅ {folder_name}: {current_count} (already balanced)")
    
    print(f"\n✅ Total files deleted: {total_deleted}")
    print(f"✅ Each folder now has {target_count} images")

# Использование
if __name__ == "__main__":
    # dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val/"
    dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_low_transform/"
    dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_low_transform_bi/"
    dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_clean/"
    
    # Вариант 1: Балансировать по минимальному количеству
    balance_dataset(dataset_path, target_count=None, keep_random=True)
    
    # Вариант 2: Установить конкретное количество (например, 500)
    # balance_dataset(dataset_path, target_count=500, keep_random=True)