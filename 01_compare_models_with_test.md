Вот переписанный код, который позволяет тестировать список моделей, сравнивать их устойчивость и выводить как отдельные графики для каждой модели, так и общие сравнительные графики в конце:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class RobustnessTester:
    """
    Тестирование устойчивости модели к различным искажениям:
    - Поворот (вращение)
    - Сдвиг (трансляция)
    - Масштабирование
    - Комбинация эффектов
    """
    
    def __init__(self, pipeline, hog, model_name, class_names, image_size=(64, 64)):
        self.pipeline = pipeline
        self.hog = hog
        self.model_name = model_name
        self.class_names = class_names
        self.image_size = image_size
        self.results = {}
    
    def apply_rotation(self, image, angle):
        """Поворот изображения"""
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=255)
        return rotated
    
    def apply_translation(self, image, dx, dy):
        """Сдвиг изображения"""
        h, w = image.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(image, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)
        return translated
    
    def apply_scale(self, image, scale_factor):
        """Масштабирование с сохранением размера"""
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        
        if scale_factor > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_h, h - new_h - pad_h,
                                       pad_w, w - new_w - pad_w,
                                       cv2.BORDER_CONSTANT, value=255)
        return scaled
    
    def extract_features_safe(self, image):
        """Безопасное извлечение HOG признаков"""
        try:
            if image.dtype != np.uint8:
                if image.max() <= 1:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size)
            
            image = np.ascontiguousarray(image)
            features = self.hog.compute(image).flatten().reshape(1, -1)
            return features
        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return np.zeros((1, self.hog.getDescriptorSize()))
    
    def test_rotation_robustness(self, X_test, y_test, angles=None, save_visualization=True):
        """Тестирование устойчивости к поворотам"""
        if angles is None:
            angles = [0, 5, 10, 15, 20, 30, 45, 60, 90, 135, 180]
        
        accuracies = []
        misclassifications = []
        
        print(f"\n🔄 [{self.model_name}] Тестирование устойчивости к ПОВОРОТАМ:")
        print("-" * 50)
        
        for angle in tqdm(angles, desc=f"   {self.model_name} - Тестирование углов"):
            transformed_images = []
            for img in X_test:
                rotated = self.apply_rotation(img, angle)
                transformed_images.append(rotated)
            
            features_list = []
            for img in transformed_images:
                feat = self.extract_features_safe(img)
                features_list.append(feat)
            
            X_transformed = np.vstack(features_list)
            y_pred = self.pipeline.predict(X_transformed)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            errors = np.sum(y_test != y_pred)
            misclassifications.append(errors)
            
            print(f"   Угол {angle:3d}°: точность = {accuracy*100:.2f}% "
                  f"(ошибок: {errors}/{len(y_test)})")
        
        # Сохраняем результаты
        self.results['rotation'] = {
            'angles': angles,
            'accuracies': accuracies,
            'misclassifications': misclassifications
        }
        
        # Визуализация
        if save_visualization:
            self.plot_robustness_curve(angles, accuracies, 
                                      f"{self.model_name} - Устойчивость к поворотам",
                                      "Угол поворота (градусы)",
                                      f"{self.model_name}_rotation_robustness.png")
        
        return angles, accuracies, misclassifications
    
    def test_translation_robustness(self, X_test, y_test, 
                                   max_shift=15, step=2,
                                   save_visualization=True):
        """Тестирование устойчивости к сдвигам"""
        shifts = list(range(0, max_shift + 1, step))
        accuracies_x = []
        accuracies_y = []
        
        print(f"\n📐 [{self.model_name}] Тестирование устойчивости к СДВИГАМ:")
        print("-" * 50)
        
        # Сдвиг по X
        print("   Сдвиг по горизонтали (X):")
        for shift in tqdm(shifts, desc=f"   {self.model_name} - Сдвиг X"):
            transformed_images = []
            for img in X_test:
                shifted = self.apply_translation(img, shift, 0)
                transformed_images.append(shifted)
            
            features_list = [self.extract_features_safe(img) for img in transformed_images]
            X_transformed = np.vstack(features_list)
            y_pred = self.pipeline.predict(X_transformed)
            accuracies_x.append(accuracy_score(y_test, y_pred))
        
        # Сдвиг по Y
        print("   Сдвиг по вертикали (Y):")
        for shift in tqdm(shifts, desc=f"   {self.model_name} - Сдвиг Y"):
            transformed_images = []
            for img in X_test:
                shifted = self.apply_translation(img, 0, shift)
                transformed_images.append(shifted)
            
            features_list = [self.extract_features_safe(img) for img in transformed_images]
            X_transformed = np.vstack(features_list)
            y_pred = self.pipeline.predict(X_transformed)
            accuracies_y.append(accuracy_score(y_test, y_pred))
        
        # Сохраняем результаты
        self.results['translation'] = {
            'shifts': shifts,
            'acc_x': accuracies_x,
            'acc_y': accuracies_y
        }
        
        # Визуализация
        if save_visualization:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(shifts, accuracies_x, 'b-o', linewidth=2, markersize=6)
            axes[0].set_xlabel('Сдвиг по X (пиксели)', fontsize=12)
            axes[0].set_ylabel('Точность', fontsize=12)
            axes[0].set_title(f'{self.model_name} - Горизонтальный сдвиг', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim([0, 1.05])
            
            axes[1].plot(shifts, accuracies_y, 'r-o', linewidth=2, markersize=6)
            axes[1].set_xlabel('Сдвиг по Y (пиксели)', fontsize=12)
            axes[1].set_ylabel('Точность', fontsize=12)
            axes[1].set_title(f'{self.model_name} - Вертикальный сдвиг', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 1.05])
            
            plt.suptitle(f'{self.model_name} - Устойчивость к сдвигам', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{self.model_name}_translation_robustness.png', dpi=150)
            plt.show()
        
        return shifts, accuracies_x, accuracies_y
    
    def test_scale_robustness(self, X_test, y_test, 
                             scale_factors=None,
                             save_visualization=True):
        """Тестирование устойчивости к масштабированию"""
        if scale_factors is None:
            scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
        
        accuracies = []
        
        print(f"\n🔍 [{self.model_name}] Тестирование устойчивости к МАСШТАБИРОВАНИЮ:")
        print("-" * 50)
        
        for scale in tqdm(scale_factors, desc=f"   {self.model_name} - Тестирование масштабов"):
            transformed_images = []
            for img in X_test:
                scaled = self.apply_scale(img, scale)
                transformed_images.append(scaled)
            
            features_list = [self.extract_features_safe(img) for img in transformed_images]
            X_transformed = np.vstack(features_list)
            y_pred = self.pipeline.predict(X_transformed)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            print(f"   Масштаб {scale:.2f}: точность = {accuracy*100:.2f}%")
        
        # Сохраняем результаты
        self.results['scale'] = {
            'scales': scale_factors,
            'accuracies': accuracies
        }
        
        if save_visualization:
            self.plot_robustness_curve(scale_factors, accuracies,
                                      f"{self.model_name} - Устойчивость к масштабированию",
                                      "Коэффициент масштабирования",
                                      f"{self.model_name}_scale_robustness.png",
                                      x_log_scale=True)
        
        return scale_factors, accuracies
    
    def test_combined_degradation(self, X_test, y_test, n_samples=100,
                                 save_visualization=True):
        """Тестирование с комбинированными искажениями"""
        print(f"\n🎯 [{self.model_name}] Тестирование с КОМБИНИРОВАННЫМИ искажениями:")
        print("-" * 50)
        
        # Создаем уровни деградации
        degradation_levels = []
        for i in range(1, 11):
            level = {
                'rotation': i * 5,
                'translation': i * 2,
                'scale': max(0.5, 1 - i * 0.05)
            }
            degradation_levels.append(level)
        
        accuracies = []
        
        for level_idx, degradation in enumerate(tqdm(degradation_levels, 
                                                     desc=f"   {self.model_name} - Тестирование")):
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_test), 
                                         min(n_samples, len(X_test)), 
                                         replace=False)
            
            correct_predictions = 0
            
            for idx in sample_idx:
                img = X_test[idx]
                true_label = y_test[idx]
                
                img_transformed = img.copy()
                img_transformed = self.apply_scale(img_transformed, degradation['scale'])
                img_transformed = self.apply_rotation(img_transformed, degradation['rotation'])
                img_transformed = self.apply_translation(img_transformed, 
                                                        degradation['translation'],
                                                        degradation['translation'])
                
                features = self.extract_features_safe(img_transformed)
                pred = self.pipeline.predict(features)[0]
                
                if pred == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(sample_idx)
            accuracies.append(accuracy)
            
            print(f"\n   Уровень {level_idx+1}:")
            print(f"      Поворот: {degradation['rotation']}°")
            print(f"      Сдвиг: {degradation['translation']}px")
            print(f"      Масштаб: {degradation['scale']:.2f}")
            print(f"      Точность: {accuracy*100:.2f}%")
        
        # Сохраняем результаты
        self.results['combined'] = {
            'levels': degradation_levels,
            'accuracies': accuracies
        }
        
        if save_visualization:
            self.plot_combined_degradation(accuracies, degradation_levels)
        
        return degradation_levels, accuracies
    
    def plot_robustness_curve(self, x_values, y_values, title, xlabel, 
                             filename, x_log_scale=False):
        """Вспомогательная функция для построения графиков"""
        plt.figure(figsize=(10, 6))
        
        if x_log_scale:
            plt.semilogx(x_values, y_values, 'g-o', linewidth=2, markersize=6)
        else:
            plt.plot(x_values, y_values, 'g-o', linewidth=2, markersize=6)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Добавляем аннотации
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            if y < 0.5:
                plt.axvline(x=x, color='red', linestyle='--', alpha=0.5)
                plt.text(x, 0.5, f'Порог: {x}', rotation=90, fontsize=8, color='red')
                break
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()
    
    def plot_combined_degradation(self, accuracies, degradation_levels):
        """Визуализация комбинированной деградации"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        levels = range(1, len(accuracies) + 1)
        
        # График точности
        axes[0, 0].plot(levels, accuracies, 'r-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Уровень деградации', fontsize=12)
        axes[0, 0].set_ylabel('Точность', fontsize=12)
        axes[0, 0].set_title(f'{self.model_name} - Падение точности при комбинированных искажениях', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # Отдельные компоненты
        rotations = [d['rotation'] for d in degradation_levels]
        translations = [d['translation'] for d in degradation_levels]
        scales = [d['scale'] for d in degradation_levels]
        
        axes[0, 1].plot(levels, rotations, 'b-o', label='Поворот (°)')
        axes[0, 1].set_xlabel('Уровень деградации', fontsize=12)
        axes[0, 1].set_ylabel('Угол поворота', fontsize=12)
        axes[0, 1].set_title('Увеличение искажений', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        ax2 = axes[0, 1].twinx()
        ax2.plot(levels, translations, 'r-s', label='Сдвиг (px)')
        ax2.set_ylabel('Сдвиг (пиксели)', fontsize=12)
        
        axes[1, 0].plot(levels, scales, 'g-^', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Уровень деградации', fontsize=12)
        axes[1, 0].set_ylabel('Коэффициент масштаба', fontsize=12)
        axes[1, 0].set_title('Уменьшение масштаба', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()
        
        # Тепловая карта ухудшения
        axes[1, 1].axis('off')
        text_summary = f"""
        📊 ИТОГИ ТЕСТИРОВАНИЯ УСТОЙЧИВОСТИ - {self.model_name}:
        
        Начальная точность: {accuracies[0]*100:.1f}%
        Финальная точность: {accuracies[-1]*100:.1f}%
        Падение точности: {(accuracies[0]-accuracies[-1])*100:.1f}%
        
        Критические уровни:
        • Точность < 80%: уровень {self._find_threshold(accuracies, 0.8)}
        • Точность < 50%: уровень {self._find_threshold(accuracies, 0.5)}
        """
        
        axes[1, 1].text(0.1, 0.5, text_summary, fontsize=10, 
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{self.model_name} - Комбинированная деградация', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_combined_degradation.png', dpi=150)
        plt.show()
    
    def _find_threshold(self, accuracies, threshold):
        """Найти уровень, на котором точность падает ниже порога"""
        for i, acc in enumerate(accuracies):
            if acc < threshold:
                return i + 1
        return len(accuracies)
    
    def run_full_robustness_test(self, X_test, y_test, save_results=True):
        """Запуск полного тестирования устойчивости"""
        print("\n" + "="*60)
        print(f"🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ УСТОЙЧИВОСТИ - {self.model_name}")
        print("="*60)
        
        # 1. Тест на повороты
        angles, rot_acc, rot_errors = self.test_rotation_robustness(X_test, y_test)
        
        # 2. Тест на сдвиги
        shifts, trans_x, trans_y = self.test_translation_robustness(X_test, y_test)
        
        # 3. Тест на масштабирование
        scales, scale_acc = self.test_scale_robustness(X_test, y_test)
        
        # 4. Комбинированный тест
        levels, combined_acc = self.test_combined_degradation(X_test, y_test)
        
        # Сохраняем результаты
        if save_results:
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """Сохранение результатов тестирования"""
        results_serializable = {
            'model_name': self.model_name,
            'rotation': {
                'angles': [float(a) for a in self.results['rotation']['angles']],
                'accuracies': [float(a) for a in self.results['rotation']['accuracies']]
            },
            'translation': {
                'shifts': [int(s) for s in self.results['translation']['shifts']],
                'acc_x': [float(a) for a in self.results['translation']['acc_x']],
                'acc_y': [float(a) for a in self.results['translation']['acc_y']]
            },
            'scale': {
                'scales': [float(s) for s in self.results['scale']['scales']],
                'accuracies': [float(a) for a in self.results['scale']['accuracies']]
            },
            'combined': {
                'accuracies': [float(a) for a in self.results['combined']['accuracies']]
            }
        }
        
        filename = f'{self.model_name}_robustness_results.json'
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\n💾 Результаты для {self.model_name} сохранены в '{filename}'")


class MultiModelRobustnessComparator:
    """
    Класс для сравнения устойчивости нескольких моделей
    """
    
    def __init__(self, models_list, X_test, y_test, class_names):
        """
        Args:
            models_list: список словарей, каждый содержит:
                - 'name': имя модели
                - 'pipeline': обученный pipeline
                - 'hog': HOG дескриптор
            X_test: тестовые изображения
            y_test: тестовые метки
            class_names: список названий классов
        """
        self.models_list = models_list
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.all_results = {}
        
    def run_all_tests(self):
        """Запуск тестов для всех моделей"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ДЛЯ ВСЕХ МОДЕЛЕЙ")
        print("="*80)
        
        for model_info in self.models_list:
            print(f"\n{'='*60}")
            print(f"📊 ТЕСТИРОВАНИЕ МОДЕЛИ: {model_info['name']}")
            print(f"{'='*60}")
            
            tester = RobustnessTester(
                pipeline=model_info['pipeline'],
                hog=model_info['hog'],
                model_name=model_info['name'],
                class_names=self.class_names
            )
            
            results = tester.run_full_robustness_test(self.X_test, self.y_test)
            self.all_results[model_info['name']] = results
            
        return self.all_results
    
    def plot_comparison_rotation(self):
        """Сравнение устойчивости к поворотам"""
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.all_results.items():
            angles = results['rotation']['angles']
            accuracies = results['rotation']['accuracies']
            plt.plot(angles, accuracies, 'o-', linewidth=2, markersize=6, label=model_name)
        
        plt.xlabel('Угол поворота (градусы)', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.title('Сравнение моделей: Устойчивость к поворотам', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('comparison_rotation.png', dpi=150)
        plt.show()
    
    def plot_comparison_translation(self):
        """Сравнение устойчивости к сдвигам"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for model_name, results in self.all_results.items():
            shifts = results['translation']['shifts']
            acc_x = results['translation']['acc_x']
            acc_y = results['translation']['acc_y']
            
            axes[0].plot(shifts, acc_x, 'o-', linewidth=2, markersize=5, label=model_name)
            axes[1].plot(shifts, acc_y, 'o-', linewidth=2, markersize=5, label=model_name)
        
        axes[0].set_xlabel('Сдвиг по X (пиксели)', fontsize=12)
        axes[0].set_ylabel('Точность', fontsize=12)
        axes[0].set_title('Сравнение: Горизонтальный сдвиг', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        axes[0].legend(loc='best')
        
        axes[1].set_xlabel('Сдвиг по Y (пиксели)', fontsize=12)
        axes[1].set_ylabel('Точность', fontsize=12)
        axes[1].set_title('Сравнение: Вертикальный сдвиг', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(loc='best')
        
        plt.suptitle('Сравнение устойчивости к сдвигам', fontsize=16)
        plt.tight_layout()
        plt.savefig('comparison_translation.png', dpi=150)
        plt.show()
    
    def plot_comparison_scale(self):
        """Сравнение устойчивости к масштабированию"""
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.all_results.items():
            scales = results['scale']['scales']
            accuracies = results['scale']['accuracies']
            plt.semilogx(scales, accuracies, 'o-', linewidth=2, markersize=6, label=model_name)
        
        plt.xlabel('Коэффициент масштабирования', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.title('Сравнение моделей: Устойчивость к масштабированию', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('comparison_scale.png', dpi=150)
        plt.show()
    
    def plot_comparison_combined(self):
        """Сравнение устойчивости к комбинированным искажениям"""
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.all_results.items():
            levels = range(1, len(results['combined']['accuracies']) + 1)
            accuracies = results['combined']['accuracies']
            plt.plot(levels, accuracies, 'o-', linewidth=2, markersize=6, label=model_name)
        
        plt.xlabel('Уровень деградации', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.title('Сравнение моделей: Комбинированные искажения', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('comparison_combined.png', dpi=150)
        plt.show()
    
    def plot_comparison_summary(self):
        """Сводный график сравнения всех метрик"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Падение точности при поворотах до 45°
        rotation_drop = {}
        for model_name, results in self.all_results.items():
            rot_acc = results['rotation']['accuracies']
            rot_angles = results['rotation']['angles']
            # Находим точность при 45°
            idx_45 = min(range(len(rot_angles)), key=lambda i: abs(rot_angles[i] - 45))
            rotation_drop[model_name] = rot_acc[0] - rot_acc[idx_45]
        
        models = list(rotation_drop.keys())
        drops = list(rotation_drop.values())
        
        bars = axes[0, 0].bar(models, drops, color='skyblue', edgecolor='navy')
        axes[0, 0].set_ylabel('Падение точности', fontsize=12)
        axes[0, 0].set_title('Падение точности при повороте на 45°', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, drop in zip(bars, drops):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{drop*100:.1f}%', ha='center', fontsize=10)
        
        # 2. Падение точности при сдвиге 10px
        translation_drop = {}
        for model_name, results in self.all_results.items():
            shifts = results['translation']['shifts']
            acc_x = results['translation']['acc_x']
            idx_10 = min(range(len(shifts)), key=lambda i: abs(shifts[i] - 10))
            translation_drop[model_name] = acc_x[0] - acc_x[idx_10]
        
        drops = list(translation_drop.values())
        bars = axes[0, 1].bar(models, drops, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_ylabel('Падение точности', fontsize=12)
        axes[0, 1].set_title('Падение точности при сдвиге на 10px', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for bar, drop in zip(bars, drops):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{drop*100:.1f}%', ha='center', fontsize=10)
        
        # 3. Падение точности при масштабировании 0.7
        scale_drop = {}
        for model_name, results in self.all_results.items():
            scales = results['scale']['scales']
            acc = results['scale']['accuracies']
            idx_07 = min(range(len(scales)), key=lambda i: abs(scales[i] - 0.7))
            scale_drop[model_name] = acc[0] - acc[idx_07]
        
        drops = list(scale_drop.values())
        bars = axes[1, 0].bar(models, drops, color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_ylabel('Падение точности', fontsize=12)
        axes[1, 0].set_title('Падение точности при масштабе 0.7', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for bar, drop in zip(bars, drops):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{drop*100:.1f}%', ha='center', fontsize=10)
        
        # 4. Общий рейтинг устойчивости
        stability_scores = {}
        for model_name, results in self.all_results.items():
            score = self.calculate_stability_score(results)
            stability_scores[model_name] = score
        
        models = list(stability_scores.keys())
        scores = list(stability_scores.values())
        
        bars = axes[1, 1].bar(models, scores, color='gold', edgecolor='orange')
        axes[1, 1].set_ylabel('Оценка устойчивости (0-10)', fontsize=12)
        axes[1, 1].set_title('Общий рейтинг устойчивости моделей', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 10])
        for bar, score in zip(bars, scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           f'{score:.1f}', ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('Сводное сравнение устойчивости моделей', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('comparison_summary.png', dpi=150)
        plt.show()
    
    def calculate_stability_score(self, results):
        """Расчет общей оценки устойчивости (0-10)"""
        score = 0
        
        # Повороты
        rot_acc = results['rotation']['accuracies']
        rot_threshold_80 = self._find_threshold(rot_acc, 0.8)
        score += min(rot_threshold_80 / 30 * 3, 3)
        
        # Сдвиги
        trans_x = results['translation']['acc_x']
        trans_threshold_90 = self._find_threshold(trans_x, 0.9)
        score += min(trans_threshold_90 / 15 * 2, 2)
        
        # Масштабирование
        scale_acc = results['scale']['accuracies']
        scale_factors = results['scale']['scales']
        min_scale_80 = min([s for s, a in zip(scale_factors, scale_acc) if a > 0.8], default=0.5)
        score += min((min_scale_80 - 0.5) / 0.5 * 2, 2)
        
        # Комбинированные
        combined_acc = results['combined']['accuracies']
        drop = combined_acc[0] - combined_acc[-1]
        score += max(0, 3 - drop * 5)
        
        return score
    
    def _find_threshold(self, accuracies, threshold):
        """Найти уровень падения точности"""
        for i, acc in enumerate(accuracies):
            if acc < threshold:
                return i + 1
        return len(accuracies)
    
    def generate_comparison_report(self):
        """Генерация полного отчета сравнения"""
        print("\n" + "="*80)
        print("📊 ИТОГОВЫЙ ОТЧЕТ СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        # Создаем таблицу результатов
        print("\n📈 СРАВНЕНИЕ ПО МЕТРИКАМ:")
        print("-" * 80)
        print(f"{'Модель':<20} {'Поворот 45°':<12} {'Сдвиг 10px':<12} {'Масштаб 0.7':<12} {'Общий рейтинг':<15}")
        print("-" * 80)
        
        for model_name, results in self.all_results.items():
            # Поворот 45°
            rot_acc = results['rotation']['accuracies']
            rot_angles = results['rotation']['angles']
            idx_45 = min(range(len(rot_angles)), key=lambda i: abs(rot_angles[i] - 45))
            rot_45 = rot_acc[idx_45]
            
            # Сдвиг 10px
            shifts = results['translation']['shifts']
            acc_x = results['translation']['acc_x']
            idx_10 = min(range(len(shifts)), key=lambda i: abs(shifts[i] - 10))
            trans_10 = acc_x[idx_10]
            
            # Масштаб 0.7
            scales = results['scale']['scales']
            acc_scale = results['scale']['accuracies']
            idx_07 = min(range(len(scales)), key=lambda i: abs(scales[i] - 0.7))
            scale_07 = acc_scale[idx_07]
            
            # Общий рейтинг
            score = self.calculate_stability_score(results)
            
            print(f"{model_name:<20} {rot_45*100:>6.1f}%       {trans_10*100:>6.1f}%       {scale_07*100:>6.1f}%       {score:>6.1f}/10")
        
        print("-" * 80)
        
        # Определяем лучшую модель
        best_model = max(self.all_results.keys(), 
                        key=lambda m: self.calculate_stability_score(self.all_results[m]))
        best_score = self.calculate_stability_score(self.all_results[best_model])
        
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ ПО УСТОЙЧИВОСТИ: {best_model} (рейтинг: {best_score:.1f}/10)")
        
        # Вывод рекомендаций
        print("\n💡 РЕКОМЕНДАЦИИ:")
        for model_name, results in self.all_results.items():
            rot_acc = results['rotation']['accuracies']
            max_angle = results['rotation']['angles'][self._find_threshold(rot_acc, 0.9)-1]
            
            trans_x = results['translation']['acc_x']
            max_shift = results['translation']['shifts'][self._find_threshold(trans_x, 0.9)-1]
            
            print(f"\n   {model_name}:")
            print(f"      • Максимальный безопасный поворот: {max_angle}°")
            print(f"      • Максимальный безопасный сдвиг: {max_shift}px")
        
        print("\n" + "="*80)
    
    def run_full_comparison(self):
        """Запуск полного сравнения всех моделей"""
        # Запускаем тесты для всех моделей
        self.run_all_tests()
        
        # Строим сравнительные графики
        print("\n" + "="*80)
        print("📊 ПОСТРОЕНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
        print("="*80)
        
        print("\n1. Сравнение устойчивости к поворотам...")
        self.plot_comparison_rotation()
        
        print("\n2. Сравнение устойчивости к сдвигам...")
        self.plot_comparison_translation()
        
        print("\n3. Сравнение устойчивости к масштабированию...")
        self.plot_comparison_scale()
        
        print("\n4. Сравнение устойчивости к комбинированным искажениям...")
        self.plot_comparison_combined()
        
        print("\n5. Сводный график сравнения...")
        self.plot_comparison_summary()
        
        # Генерируем отчет
        self.generate_comparison_report()
        
        return self.all_results


# ============================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================

def main_with_multiple_models():
    """
    Пример использования для тестирования нескольких моделей
    """
    # Здесь должна быть ваша логика загрузки данных
    # X_test, y_test, class_names = prepare_test_data(...)
    
    # Пример: список моделей для тестирования
    models_to_test = [
        {
            'name': 'SVM_RBF',
            'pipeline': pipeline_1,  # ваш pipeline
            'hog': hog_1             # ваш HOG дескриптор
        },
        {
            'name': 'SVM_LINEAR',
            'pipeline': pipeline_2,
            'hog': hog_2
        },
        {
            'name': 'RandomForest',
            'pipeline': pipeline_3,
            'hog': hog_3
        }
    ]
    
    # Создаем компаратор
    comparator = MultiModelRobustnessComparator(
        models_list=models_to_test,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names
    )
    
    # Запускаем полное сравнение
    all_results = comparator.run_full_comparison()
    
    return all_results


if __name__ == "__main__":
    # Запуск тестирования для списка моделей
    all_results = main_with_multiple_models()
```

## Основные изменения:

### 1. **Класс `RobustnessTester`** 
- Добавлен параметр `model_name` для идентификации модели
- Результаты сохраняются с именем модели в файлах
- Отдельные графики для каждой модели

### 2. **Новый класс `MultiModelRobustnessComparator`**
- Принимает список моделей для тестирования
- Запускает тесты для всех моделей
- Строит сравнительные графики:
  - Сравнение поворотов
  - Сравнение сдвигов
  - Сравнение масштабирования
  - Сравнение комбинированных искажений
  - Сводный график со всеми метриками

### 3. **Функции сравнения**:
- `plot_comparison_rotation()` - сравнение поворотов
- `plot_comparison_translation()` - сравнение сдвигов
- `plot_comparison_scale()` - сравнение масштабирования
- `plot_comparison_combined()` - сравнение комбинированных искажений
- `plot_comparison_summary()` - сводный график
- `generate_comparison_report()` - текстовый отчет

### 4. **Результаты**:
- Для каждой модели создаются отдельные графики
- В конце создаются общие сравнительные графики
- Генерируется итоговый отчет с рейтингом моделей

### Использование:

```python
# Создайте список ваших моделей
models = [
    {'name': 'Model1', 'pipeline': pipe1, 'hog': hog1},
    {'name': 'Model2', 'pipeline': pipe2, 'hog': hog2},
    # ... добавьте больше моделей
]

# Запустите сравнение
comparator = MultiModelRobustnessComparator(models, X_test, y_test, class_names)
results = comparator.run_full_comparison()
```