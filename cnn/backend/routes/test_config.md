curl -X GET http://localhost:5001/config | jq '.'





















Отличный код! Вот полный набор команд curl для тестирования всех эндпоинтов вашего API:

## 1. Базовые команды для тестирования

### Получить все настройки (GET)
```bash
# GET все настройки
curl -X GET http://localhost:5000/config

# С красивым форматированием (если установлен jq)
curl -X GET http://localhost:5000/config | jq '.'

# Сохранить результат в файл
curl -X GET http://localhost:5000/config -o config_response.json
```

### Получить конкретный параметр
```bash
# Получить URL камеры
curl -X GET http://localhost:5000/config/camera_url

# Получить количество цифр
curl -X GET http://localhost:5000/config/digit_count

# Получить порог сохранения
curl -X GET http://localhost:5000/config/save_threshold

# Получить настройки обрезки
curl -X GET http://localhost:5000/config/crop_top
```

## 2. Обновление настроек (POST)

### Обновить один параметр
```bash
# Обновить URL камеры
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"camera_url": "rtsp://192.168.1.100:554/stream"}'

# Обновить количество цифр
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"digit_count": 6}'

# Включить режим отладки
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"debug_mode": true}'
```

### Обновить несколько параметров одновременно
```bash
# Обновить параметры камеры
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "camera_url": "rtsp://admin:password@192.168.1.100:554/stream",
    "camera_request_pause": 3,
    "monitoring_enabled": true
  }'

# Обновить параметры обрезки
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "crop_top": 150,
    "crop_left": 80,
    "crop_right": 10,
    "crop_bottom": 20
  }'

# Обновить параметры распознавания
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "digit_width": 32,
    "digit_count": 5,
    "save_threshold": 0.7
  }'

# Полное обновление всех настроек
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "camera_url": "rtsp://192.168.1.100:554/stream",
    "camera_request_pause": 2,
    "monitoring_enabled": true,
    "save_threshold": 0.6,
    "save_bad_photos": true,
    "crop_top": 120,
    "crop_left": 60,
    "crop_right": 0,
    "crop_bottom": 0,
    "digit_width": 28,
    "digit_count": 5,
    "debug_mode": false
  }'
```

## 3. Обновление через PUT (альтернативный метод)

```bash
# PUT запрос для обновления параметра
curl -X PUT http://localhost:5000/config/camera_url \
  -H "Content-Type: application/json" \
  -d '{"value": "rtsp://new_camera:554/stream"}'

curl -X PUT http://localhost:5000/config/digit_count \
  -H "Content-Type: application/json" \
  -d '{"value": 6}'

curl -X PUT http://localhost:5000/config/crop_top \
  -H "Content-Type: application/json" \
  -d '{"value": 200}'
```

## 4. Сброс настроек

```bash
# Сброс к значениям по умолчанию
curl -X POST http://localhost:5000/config/reset

# Проверить, что сбросилось
curl -X GET http://localhost:5000/config
```

## 5. Тестирование сценариев

### Сценарий 1: Настройка новой камеры
```bash
# 1. Проверить текущие настройки
curl -X GET http://localhost:5000/config

# 2. Настроить камеру
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "camera_url": "rtsp://admin:12345@192.168.1.100:554/stream",
    "camera_request_pause": 5
  }'

# 3. Проверить, что применилось
curl -X GET http://localhost:5000/config/camera_url
```

### Сценарий 2: Оптимизация распознавания
```bash
# Увеличить чувствительность (меньше порог)
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "save_threshold": 0.3,
    "save_bad_photos": true
  }'

# Изменить параметры цифр
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "digit_width": 32,
    "digit_count": 6
  }'
```

### Сценарий 3: Настройка области распознавания
```bash
# Установить область обрезки
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "crop_top": 100,
    "crop_left": 50,
    "crop_right": 20,
    "crop_bottom": 30
  }'
```

## 6. Тестирование с переменными (для удобства)

```bash
# Сохранить URL в переменную
API_URL="http://localhost:5000"

# Использовать переменную
curl -X GET ${API_URL}/config
curl -X GET ${API_URL}/config/digit_count

# Обновление с переменными
curl -X POST ${API_URL}/config \
  -H "Content-Type: application/json" \
  -d "{\"camera_url\": \"rtsp://192.168.1.100:554/stream\"}"
```

## 7. Полезные опции curl

```bash
# Показать заголовки ответа
curl -i -X GET http://localhost:5000/config

# Показать только статус
curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/config

# Измерить время ответа
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/config

# Создать файл curl-format.txt со следующим содержимым:
#     time_namelookup:  %{time_namelookup}s\n
#        time_connect:  %{time_connect}s\n
#     time_appconnect:  %{time_appconnect}s\n
#    time_pretransfer:  %{time_pretransfer}s\n
#       time_redirect:  %{time_redirect}s\n
#  time_starttransfer:  %{time_starttransfer}s\n
#                     ----------\n
#          time_total:  %{time_total}s\n
```

## 8. Batch-тестирование (bash скрипт)

```bash
#!/bin/bash
# test_config_api.sh

API_URL="http://localhost:5000"
echo "=== Тестирование Config API ===="

# 1. Получить все настройки
echo -e "\n1. Получение всех настроек:"
curl -s -X GET ${API_URL}/config | jq '.'

# 2. Получить конкретный параметр
echo -e "\n2. Получение camera_url:"
curl -s -X GET ${API_URL}/config/camera_url | jq '.'

# 3. Обновить настройки
echo -e "\n3. Обновление настроек:"
curl -s -X POST ${API_URL}/config \
  -H "Content-Type: application/json" \
  -d '{
    "camera_request_pause": 3,
    "debug_mode": true
  }' | jq '.'

# 4. Проверить, что обновилось
echo -e "\n4. Проверка обновлений:"
curl -s -X GET ${API_URL}/config | jq '.config | {camera_request_pause, debug_mode}'

# 5. Обновить параметр через PUT
echo -e "\n5. Обновление через PUT:"
curl -s -X PUT ${API_URL}/config/digit_count \
  -H "Content-Type: application/json" \
  -d '{"value": 6}' | jq '.'

# 6. Сброс настроек
echo -e "\n6. Сброс настроек:"
curl -s -X POST ${API_URL}/config/reset | jq '.'

echo -e "\n=== Тестирование завершено ==="
```

## 9. Тестирование ошибок

```bash
# Несуществующий параметр
curl -X GET http://localhost:5000/config/non_existent_param

# Неверный формат запроса (без поля value)
curl -X PUT http://localhost:5000/config/camera_url \
  -H "Content-Type: application/json" \
  -d '{"wrong_field": "value"}'

# Пустой запрос
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{}'

# Неверный тип данных
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"digit_count": "not_a_number"}'
```

## 10. Мониторинг в реальном времени

```bash
# Бесконечный мониторинг изменений (каждые 2 секунды)
watch -n 2 'curl -s http://localhost:5000/config | jq ".config | {camera_url, digit_count, crop_top}"'

# Логирование изменений
while true; do
  echo "=== $(date) ===" >> config_log.txt
  curl -s http://localhost:5000/config >> config_log.txt
  echo "" >> config_log.txt
  sleep 5
done
```

## 11. Использование с jq для фильтрации

```bash
# Установить jq если нет: sudo apt-get install jq (Linux) или brew install jq (Mac)

# Получить только определенные поля
curl -s http://localhost:5000/config | jq '.config | {camera_url, digit_count, save_threshold}'

# Получить все настройки камеры
curl -s http://localhost:5000/config | jq '.config | with_entries(select(.key | contains("camera")))'

# Получить числовые настройки
curl -s http://localhost:5000/config | jq '.config | with_entries(select(.value | type == "number"))'

# Проверить успешность обновления
curl -s -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"debug_mode": true}' | jq '.status'
```

## 12. Тест-кейс для автоматизации

```python
# test_api.py - Python скрипт для автоматического тестирования
import requests
import json

API_URL = "http://localhost:5000"

def test_get_config():
    """Тест получения конфигурации"""
    response = requests.get(f"{API_URL}/config")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "config" in data
    print("✓ GET /config - OK")

def test_update_config():
    """Тест обновления конфигурации"""
    updates = {
        "camera_request_pause": 10,
        "debug_mode": True
    }
    response = requests.post(f"{API_URL}/config", json=updates)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    print("✓ POST /config - OK")

def test_reset_config():
    """Тест сброса конфигурации"""
    response = requests.post(f"{API_URL}/config/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    print("✓ POST /config/reset - OK")

if __name__ == "__main__":
    test_get_config()
    test_update_config()
    test_reset_config()
    print("\n✅ Все тесты пройдены!")
```

Запускайте эти команды после запуска вашего Flask приложения. Все команды предполагают, что приложение работает на `http://localhost:5000`. Если порт другой, замените в командах.