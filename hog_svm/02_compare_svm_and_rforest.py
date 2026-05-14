import os
import numpy as np
import cv2
# Используем агрессивный backend для предотвращения проблем с потоками
import matplotlib
matplotlib.use('Agg')  # <-- ВАЖНО: добавить ДО импорта pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import torch
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings('ignore')

# Импорт ваших кастомных трансформаций
from augmentation import AdaptiveAugmentationBuilder, CenterDigitsTransform, SquarePad
from transform_helper import ExtractLetterWithMargin

DATA_ROOT = "/mnt/ntfs/learn_ML/torch_gas_counter/prog_3/dataset"

# ============================================
# 1. HOG DESCRIPTOR
# ============================================

def create_hog_descriptor(image_size=(64, 64)):
    """Создание HOG дескриптора"""
    win_size = image_size
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )
    return hog


def extract_hog_features(images, hog=None, image_size=(64, 64), verbose=True):
    """
    Extract HOG features from images.
    
    Args:
        images: List of numpy arrays
        hog: HOG descriptor (creates new if None)
        image_size: Target size (width, height)
        verbose: Print progress
    
    Returns:
        features: numpy array of shape (n_samples, feature_dim)
    """
    if hog is None:
        hog = create_hog_descriptor(image_size)
    
    n_samples = len(images)
    feature_dim = hog.getDescriptorSize()
    
    if verbose:
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Processing {n_samples} images...")
    
    features = np.zeros((n_samples, feature_dim), dtype=np.float32)
    
    for i in range(n_samples):
        img = images[i]
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Ensure 2D grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 1:
                img = img.squeeze()
        
        # Resize if needed
        if img.shape != image_size:
            img = cv2.resize(img, image_size)
        
        # Ensure contiguous array (fixes some OpenCV issues)
        img = np.ascontiguousarray(img)
        
        # Compute HOG features
        try:
            feat = hog.compute(img)
            features[i] = feat.flatten()
        except Exception as e:
            if verbose and i < 5:
                print(f"   Warning: Error on image {i}: {e}")
            features[i] = np.zeros(feature_dim)
        
        # Progress indicator
        if verbose and (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{n_samples} images...")
    
    if verbose:
        print(f"   Done! Features shape: {features.shape}")
    
    return features


# ============================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================

def load_config(config_path='config.yaml'):
    """Loads the configuration"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {'data': {'image_size': 64}}


def load_and_split_data(data_root, test_size=0.15, val_size=0.15):
    """
    Загружает данные с базовым трансформом и разделяет на train/val/test.
    Возвращает numpy массивы изображений и меток.
    """
    print("\n📂 Загрузка данных...")
    
    # Базовый трансформ для всех данных (без аугментаций)
    base_transform = transforms.Compose([
        CenterDigitsTransform(padding=2, fill_value=0),
        SquarePad(fill_white=False),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    # Загружаем датасет
    full_dataset = datasets.ImageFolder(root=data_root, transform=base_transform)
    class_names = full_dataset.classes
    
    # Конвертируем в numpy
    all_images = []
    all_labels = []
    
    print("   Конвертация изображений в numpy...")
    for i in tqdm(range(len(full_dataset)), desc="   Загрузка"):
        img_tensor, label = full_dataset[i]
        img_np = img_tensor.squeeze().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

        all_images.append(img_np)
        all_labels.append(label)
    
    X = np.stack(all_images)
    y = np.array(all_labels)
    
    print(f"   Всего изображений: {len(X)}")
    print(f"   Классов: {len(class_names)}")
    print(f"   Форма изображений: {X[0].shape}")
    
    # Разделение на train/val/test
    print("\n✂️ Стратифицированное разделение на train/val/test...")
    
    # Сначала разделяем на train (70%) и temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )
    
    # Затем разделяем temp на val (15%) и test (15%)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=42
    )
    
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def apply_augmentations_to_images(images, aug_transform):
    """
    Применяет аугментации к numpy изображениям.
    
    Args:
        images: numpy array of images (shape: n_samples, height, width)
        aug_transform: torchvision transform with augmentations
    
    Returns:
        augmented_images: numpy array of augmented images
    """
    print("   Применение аугментаций к train изображениям...")
    augmented_images = []
    
    for i in tqdm(range(len(images)), desc="   Аугментация"):
        img = images[i]
        # Конвертируем numpy в PIL
        img_pil = Image.fromarray(img)
        # Применяем аугментации
        img_aug = aug_transform(img_pil)
        # Конвертируем обратно в numpy
        img_np = img_aug.squeeze().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        augmented_images.append(img_np)
    
    return np.stack(augmented_images)


# ============================================
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

def train_svm(X_train, y_train, X_val, y_val):
    """Обучение SVM с подбором гиперпараметров"""
    print("\n🏋️ ОБУЧЕНИЕ SVM...")
    print("-" * 40)
    
    # Создаем pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('svm', svm.SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    
    # Подбор гиперпараметров
    param_grid = {
        'svm__C': [10],
        'svm__gamma': [0.01, 0.001],
        'svm__kernel': ['rbf']
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Оценка
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    overfit_gap = train_acc - val_acc
    
    print(f"\n   Лучшие параметры: {grid_search.best_params_}")
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Val Accuracy: {val_acc*100:.2f}%")
    print(f"   Разрыв train-val: {overfit_gap*100:.2f}%")
    
    if overfit_gap > 0.03:
        print(f"   ⚠️ ВНИМАНИЕ: Возможно переобучение! (разрыв > 3%)")
    
    # PCA explained variance
    if 'pca' in best_model.named_steps:
        pca = best_model.named_steps['pca']
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"   PCA explained variance: {explained_var:.2%}")
    
    return best_model, train_acc, val_acc, grid_search.best_params_


def train_random_forest(X_train, y_train, X_val, y_val):
    """Обучение Random Forest с подбором гиперпараметров"""
    print("\n🌲 ОБУЧЕНИЕ RANDOM FOREST...")
    print("-" * 40)
    
    # Создаем pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    
    # Подбор гиперпараметров
    param_grid = {
        'rf__n_estimators': [300],
        'rf__max_depth': [10, 20],
        'rf__min_samples_split': [5, 10],
        'rf__min_samples_leaf': [2, 4, 8]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Оценка
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    # Проверка переобучения
    overfit_gap = train_acc - val_acc
    
    print(f"\n   Лучшие параметры: {grid_search.best_params_}")
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Val Accuracy: {val_acc*100:.2f}%")
    print(f"   Разрыв train-val: {overfit_gap*100:.2f}%")
    
    if overfit_gap > 0.03:
        print(f"   ⚠️ ВНИМАНИЕ: Сильное переобучение! (разрыв > 3%)")
    
    # Важность признаков
    rf = best_model.named_steps['rf']
    feature_importance = rf.feature_importances_
    top5_idx = np.argsort(feature_importance)[-5:][::-1]
    print(f"   Важность признаков (top 5): {feature_importance[top5_idx]}")
    
    return best_model, train_acc, val_acc, grid_search.best_params_


# ============================================
# 4. ОЦЕНКА И СРАВНЕНИЕ
# ============================================

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """Оценка модели на тестовой выборке"""
    print(f"\n📊 ОЦЕНКА {model_name} НА ТЕСТОВОЙ ВЫБОРКЕ:")
    print("-" * 50)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"   Macro avg - Precision: {report['macro avg']['precision']:.3f}")
    print(f"   Macro avg - Recall: {report['macro avg']['recall']:.3f}")
    print(f"   Macro avg - F1: {report['macro avg']['f1-score']:.3f}")
    
    return accuracy, report, y_pred


def compare_models(svm_model, rf_model, X_test, y_test, class_names):
    """Сравнение SVM и Random Forest"""
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    # Оценка обеих моделей
    svm_acc, svm_report, svm_pred = evaluate_model(svm_model, X_test, y_test, class_names, "SVM")
    rf_acc, rf_report, rf_pred = evaluate_model(rf_model, X_test, y_test, class_names, "Random Forest")
    
    # Визуализация сравнения
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Сравнение точности
    models = ['SVM', 'Random Forest']
    accuracies = [svm_acc, rf_acc]
    colors = ['#2E86AB', '#A23B72']
    
    axes[0, 0].bar(models, accuracies, color=colors, edgecolor='black')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Сравнение точности', fontsize=14)
    axes[0, 0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% порог')
    axes[0, 0].legend()
    
    # Добавляем значения на столбцы
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc*100:.1f}%', ha='center', fontsize=12)
    
    # 2. Сравнение метрик
    metrics = ['Precision', 'Recall', 'F1-score']
    svm_scores = [svm_report['macro avg']['precision'], 
                  svm_report['macro avg']['recall'],
                  svm_report['macro avg']['f1-score']]
    rf_scores = [rf_report['macro avg']['precision'],
                 rf_report['macro avg']['recall'],
                 rf_report['macro avg']['f1-score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, svm_scores, width, label='SVM', color=colors[0])
    axes[0, 1].bar(x + width/2, rf_scores, width, label='Random Forest', color=colors[1])
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Сравнение метрик (macro avg)', fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Confusion Matrix для SVM
    cm_svm = confusion_matrix(y_test, svm_pred)
    im = axes[0, 2].imshow(cm_svm, cmap='Blues')
    axes[0, 2].set_title('Confusion Matrix - SVM', fontsize=14)
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('True')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. Confusion Matrix для Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    im = axes[1, 0].imshow(cm_rf, cmap='Greens')
    axes[1, 0].set_title('Confusion Matrix - Random Forest', fontsize=14)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Разница в предсказаниях
    different_predictions = svm_pred != rf_pred
    n_different = np.sum(different_predictions)
    
    # 6. Анализ где какая модель лучше
    svm_correct = (svm_pred == y_test)
    rf_correct = (rf_pred == y_test)
    
    both_correct = svm_correct & rf_correct
    both_wrong = ~svm_correct & ~rf_correct
    svm_only_correct = svm_correct & ~rf_correct
    rf_only_correct = ~svm_correct & rf_correct
    
    labels_both = ['Both Correct', 'Both Wrong', 'Only SVM', 'Only RF']
    counts = [np.sum(both_correct), np.sum(both_wrong), 
              np.sum(svm_only_correct), np.sum(rf_only_correct)]
    
    axes[1, 1].bar(labels_both, counts, color=['green', 'red', 'blue', 'orange'])
    axes[1, 1].set_ylabel('Number of samples', fontsize=12)
    axes[1, 1].set_title('Сравнение предсказаний', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Добавляем значения
    for i, v in enumerate(counts):
        axes[1, 1].text(i, v + 5, str(v), ha='center', fontsize=10)
    
    # 7. Статистика
    axes[1, 2].axis('off')
    
    diff_percent = (n_different / len(y_test)) * 100
    svm_better_percent = (np.sum(svm_only_correct) / len(y_test)) * 100
    rf_better_percent = (np.sum(rf_only_correct) / len(y_test)) * 100
    
    summary_text = f"""
    📊 СТАТИСТИКА СРАВНЕНИЯ:
    
    • Модели предсказали по-разному: 
      {n_different}/{len(y_test)} ({diff_percent:.1f}%)
    
    • SVM лучше Random Forest:
      {np.sum(svm_only_correct)} примеров ({svm_better_percent:.1f}%)
    
    • Random Forest лучше SVM:
      {np.sum(rf_only_correct)} примеров ({rf_better_percent:.1f}%)
    
    🏆 ПОБЕДИТЕЛЬ:
      {'SVM' if svm_acc > rf_acc else 'Random Forest'}
      (точность: {max(svm_acc, rf_acc)*100:.1f}%)
    
    💡 РЕКОМЕНДАЦИЯ:
      Использовать {'SVM' if svm_acc > rf_acc else 'Random Forest'}
      для HOG признаков.
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('HOG + SVM vs HOG + Random Forest', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_vs_rf_comparison.png', dpi=150)
    plt.show()
    
    # Сохраняем результаты
    results = {
        'svm': {'accuracy': svm_acc, 'report': svm_report, 'predictions': svm_pred},
        'rf': {'accuracy': rf_acc, 'report': rf_report, 'predictions': rf_pred},
        'comparison': {
            'different_predictions': n_different,
            'svm_only_correct': int(np.sum(svm_only_correct)),
            'rf_only_correct': int(np.sum(rf_only_correct)),
            'both_correct': int(np.sum(both_correct)),
            'both_wrong': int(np.sum(both_wrong)),
            'winner': 'SVM' if svm_acc > rf_acc else 'Random Forest'
        }
    }
    
    return results


def save_models(svm_model, rf_model, class_names, svm_acc, rf_acc, test_acc_svm, test_acc_rf, hog):
    """Сохранение моделей"""
    print("\n💾 Сохранение моделей...")
    
    # Параметры HOG для сохранения
    hog_params = {
        'win_size': (64, 64),
        'block_size': (16, 16),
        'block_stride': (8, 8),
        'cell_size': (8, 8),
        'nbins': 9
    }

    # Сохраняем SVM
    svm_filename = 'hog_svm_model.pkl'
    joblib.dump({
        'pipeline': svm_model,
        'class_names': class_names,
        'train_accuracy': svm_acc,
        'test_accuracy': test_acc_svm,
        'hog_params': hog_params
    }, svm_filename)
    print(f"   ✓ SVM сохранена: {svm_filename}")
    
    # Сохраняем Random Forest
    rf_filename = 'hog_rf_model.pkl'
    joblib.dump({
        'pipeline': rf_model,
        'class_names': class_names,
        'train_accuracy': rf_acc,
        'test_accuracy': test_acc_rf,
        'hog_params': hog_params
    }, rf_filename)
    print(f"   ✓ Random Forest сохранена: {rf_filename}")


# ============================================
# 5. MAIN
# ============================================

def main():
    """Основная функция"""
    print("🔧 СРАВНЕНИЕ HOG + SVM vs HOG + Random Forest")
    print("="*70)
    
    # 1. Загрузка и разделение данных (без аугментаций)
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_and_split_data(
        DATA_ROOT, test_size=0.15, val_size=0.15
    )
    
    # 2. Создание аугментаций для train
    print("\n🔧 Создание аугментаций для train данных...")
    config = load_config()
    aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])
    
    train_transform = aug_builder.build_train_image_transform((64, 64))
    
    # 3. Применяем аугментации ТОЛЬКО к train данным
    X_train_aug = apply_augmentations_to_images(X_train, train_transform)
    
    # 4. Извлечение HOG признаков
    print("\n🔧 Извлечение HOG признаков...")
    hog = create_hog_descriptor(image_size=(64, 64))
    
    print("\n   Training set (with augmentations):")
    X_train_hog = extract_hog_features(X_train_aug, hog, verbose=True)
    
    print("\n   Validation set (without augmentations):")
    X_val_hog = extract_hog_features(X_val, hog, verbose=True)
    
    print("\n   Test set (without augmentations):")
    X_test_hog = extract_hog_features(X_test, hog, verbose=True)
    
    # 5. Обучение SVM
    svm_model, svm_train_acc, svm_val_acc, svm_params = train_svm(
        X_train_hog, y_train, X_val_hog, y_val
    )
    
    # 6. Обучение Random Forest
    rf_model, rf_train_acc, rf_val_acc, rf_params = train_random_forest(
        X_train_hog, y_train, X_val_hog, y_val
    )
    
    # 7. Сравнение моделей на тестовой выборке
    comparison_results = compare_models(
        svm_model, rf_model, X_test_hog, y_test, class_names
    )
    
    # 8. Сохранение моделей
    save_models(
        svm_model, rf_model, class_names,
        svm_train_acc, rf_train_acc,
        comparison_results['svm']['accuracy'],
        comparison_results['rf']['accuracy'],
        hog
    )
    
    # 9. Итоговый вывод
    print("\n" + "="*70)
    print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("="*70)
    print(f"\n🏆 ПОБЕДИТЕЛЬ: {comparison_results['comparison']['winner']}")
    print(f"   SVM точность: {comparison_results['svm']['accuracy']*100:.2f}%")
    print(f"   Random Forest точность: {comparison_results['rf']['accuracy']*100:.2f}%")
    
    return svm_model, rf_model, class_names, comparison_results


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    svm_model, rf_model, class_names, results = main()