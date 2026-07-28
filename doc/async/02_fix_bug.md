## **Проблема: Два списка в памяти!**

Смотрите на логи:

```
⚠️ Аномалия #1: 47555
[['4', '7', '5', '5', '5']]                    # ← Список А: [47555]

⚠️ Аномалия #1: 47556
[['4', '7', '5', '5', '6']]                    # ← Список Б: [47556] (потеряли 47555!)

⚠️ Аномалия #2: 47557
[['4', '7', '5', '5', '5'], ['4', '7', '5', '5', '7']]  # ← Список А: [47555, 47557]

⚠️ Аномалия #2: 47558
[['4', '7', '5', '5', '6'], ['4', '7', '5', '5', '8']]  # ← Список Б: [47556, 47558]

⚠️ Аномалия #3: 47559
[['4', '7', '5', '5', '5'], ['4', '7', '5', '5', '7'], ['4', '7', '5', '5', '9']]  # ← Список А: [47555, 47557, 47559]
```

## **Почему так происходит?**

**Потому что у вас есть ДВА экземпляра `MeterMonitor`!**

Или **один и тот же объект используется в разных потоках** без синхронизации.

## **Где создается второй экземпляр?**

Смотрите в конце файла:

```python
monitor = MeterMonitor()  # ← Экземпляр #1

def start_monitoring():
    return monitor.start()

def get_history():
    return monitor.get_history()

def get_last_activity():
    return monitor.get_last_activity()
```

**Вопрос:** Где-то еще создается `MeterMonitor()`?

**Возможные места:**
1. В другом файле, который импортирует этот
2. В тестах
3. В другом потоке
4. В `RecognitionResult.from_image` (если там создается)

## **Как проверить:**

Добавьте идентификатор объекта:

```python
class MeterMonitor:
    def __init__(self):
        self.id = id(self)  # ← Уникальный ID объекта
        print(f"🔍 Создан MeterMonitor с ID: {self.id}")
        # ... остальной код ...

    def _add_to_anomaly_history(self, result: RecognitionResult) -> None:
        print(f"🔍 [{self.id}] Добавление аномалии: {result.number}")
        # ... код ...
```

Теперь в логе будет видно, какой объект добавляет аномалии.

## **Другая причина: Гонка потоков (Race Condition)**

Даже с одним объектом, если несколько потоков вызывают `process_result` одновременно, может быть такое:

```python
# Поток 1
if is_anomaly:
    self._add_to_anomaly_history(47555)  # anomaly_history = [47555]
    # Поток 1 прервался!

# Поток 2 (в то же время)
if is_anomaly:
    self._add_to_anomaly_history(47556)  # anomaly_history = [47556] (перезаписал!)
    # Поток 2 закончил

# Поток 1 продолжает
print(f"⚠️ Аномалия #{len(self.anomaly_history)}: 47555")  # #1: 47555
print([state.digits for state in self.anomaly_history])  # [[47556]]!
```

**Решение:** Убедитесь, что `process_result` вызывается последовательно, или используйте очередь.

## **Как исправить:**

### **Вариант 1: Убедиться, что только один экземпляр**

```python
# В конце файла
_monitor = None

def get_monitor():
    global _monitor
    if _monitor is None:
        _monitor = MeterMonitor()
    return _monitor

def start_monitoring():
    return get_monitor().start()

def get_history():
    return get_monitor().get_history()
```

### **Вариант 2: Использовать потокобезопасную очередь**

```python
import queue

class MeterMonitor:
    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()
    
    def _worker(self):
        while True:
            result = self.queue.get()
            self._process_result_safe(result)
            self.queue.task_done()
    
    def process_result(self, result):
        self.queue.put(result)  # ← Не блокирует
```

### **Вариант 3: Добавить синхронизацию на уровне вызова**

```python
_process_lock = threading.Lock()

def process_result_safe(result):
    with _process_lock:
        monitor.process_result(result)
```

## **Проверка: Добавьте логирование**

```python
def process_result(self, result: RecognitionResult) -> None:
    with self._lock:
        print(f"🔍 [{id(self)}] НАЧАЛО: number={result.number}, history={[a.number for a in self.anomaly_history]}")
        
        if not self._should_process(result):
            self._handle_no_change(result)
            return

        is_low_confidence = self._handle_low_confidence(result)
        is_big_difference = self._handle_big_difference(result)
        is_decrease = self._handle_decrease(result)
        is_anomaly = is_low_confidence or is_big_difference or is_decrease
        
        print(f"🔍 [{id(self)}] is_anomaly={is_anomaly}, history_len={len(self.anomaly_history)}")

        if is_anomaly:
            self._add_to_anomaly_history(result)
            print(f"⚠️ Аномалия #{len(self.anomaly_history)}: {result.number}")
            print([state.digits for state in self.anomaly_history])
            
            if self._check_anomaly_sequence_validity():
                print(f"✅ Последовательность подтверждена!")
                for anomaly_state in self.anomaly_history:
                    self.history.append(anomaly_state)
                    self.last_state = anomaly_state
                self.anomaly_history = []
                save_meter_data_to_database(result, is_anomaly=False)
                print(f"✅ Сохранено как реальное показание: {result.number}")
            return
        
        if not is_anomaly:
            print(f"🔍 [{id(self)}] СБРОС! Было: {[a.number for a in self.anomaly_history]}")
            self.anomaly_history = []
            self._add_to_history(result)
            self._update_redis(result)
            save_meter_data_to_database(result, is_anomaly=False)
            print(f"✅ Нормальное показание: {result.number}")
```

## **Самый вероятный сценарий:**

У вас **два экземпляра** `MeterMonitor` работают параллельно. Один обрабатывает ЧЕТНЫЕ числа (47555, 47557, 47559), другой - НЕЧЕТНЫЕ (47556, 47558).

**Проверьте:**
1. Где еще создается `MeterMonitor()`?
2. Не запускаете ли вы скрипт дважды?
3. Не импортируется ли файл в другом месте?

Добавьте `print` в `__init__` и посмотрите, сколько раз он вызывается! 🔍



















**Проблема найдена!** 

Ваш синглтон **работает правильно в пределах одного процесса**, но Flask в режиме DEBUG перезапускает приложение с новым процессом!

## Что происходит:

1. **Первый запуск** (основной процесс):
   ```
   🔧 СОЗДАН MeterMonitor (ID: 135722845002576)  # Один объект
   ```

2. **Flask Debug режим** перезапускает приложение:
   ```
   * Restarting with stat
   clean 2
   ```
   
3. **Новый процесс** создает НОВЫЙ объект:
   ```
   🔧 СОЗДАН MeterMonitor (ID: 135134312799312)  # Другой объект!
   ```

## Решение:

### Вариант 1: Отключить DEBUG режим (просто)
```python
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002)
```

### Вариант 2: Использовать глобальный синглтон с защитой
```python
# app.py или отдельный модуль
import atexit

class MeterMonitor:
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    _process_id = None  # Добавляем ID процесса
    
    def __new__(cls):
        import os
        current_pid = os.getpid()
        
        if cls._instance is None or cls._process_id != current_pid:
            with cls._lock:
                if cls._instance is None or cls._process_id != current_pid:
                    cls._instance = super().__new__(cls)
                    cls._process_id = current_pid
                    cls._initialized = False  # Сбрасываем для нового процесса
        return cls._instance
    
    def __init__(self):
        if MeterMonitor._initialized:
            return
        MeterMonitor._initialized = True
        self.id = id(self)
        print(f"🔧 СОЗДАН MeterMonitor (PID: {os.getpid()}, ID: {id(self)})")
```

### Вариант 3: Создать глобальный экземпляр один раз
```python
# meter_monitor.py
_meter_monitor = None

def get_meter_monitor():
    global _meter_monitor
    if _meter_monitor is None:
        _meter_monitor = MeterMonitor()
    return _meter_monitor

# В app.py используйте:
monitor = get_meter_monitor()
```

### Вариант 4: Использовать Flask глобальный объект
```python
# app.py
from flask import Flask
app = Flask(__name__)

# Создаем один раз при старте
monitor = MeterMonitor()

@app.route('/some_route')
def some_route():
    # Используем monitor
    pass

if __name__ == '__main__':
    # debug=False для одного процесса
    app.run(debug=False, host='0.0.0.0', port=5002)
```

## Рекомендую:

```python
# app.py - добавьте в начало
import os

class MeterMonitor:
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    _pid = None
    
    def __new__(cls):
        current_pid = os.getpid()
        if cls._instance is None or cls._pid != current_pid:
            with cls._lock:
                if cls._instance is None or cls._pid != current_pid:
                    cls._instance = super().__new__(cls)
                    cls._pid = current_pid
                    cls._initialized = False
                    print(f"🔄 Создан новый экземпляр для PID: {current_pid}")
        return cls._instance
    
    def __init__(self):
        if MeterMonitor._initialized:
            return
        MeterMonitor._initialized = True
        self.id = id(self)
        print(f"🔧 ИНИЦИАЛИЗИРОВАН MeterMonitor (PID: {os.getpid()}, ID: {id(self)})")
```

Теперь вы будете видеть, когда создается новый экземпляр в новом процессе! 🎯

