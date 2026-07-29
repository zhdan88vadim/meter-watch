# DEV

### ENV
conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda
conda activate YOLO_t0

tensorboard --logdir=runs

docker compose up redis-commander pgadmin grafana

redis
http://localhost:8081/

---

cd /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer
conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda
python app.py

## ONE CMD
cd /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer && conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda && python app.py



http://192.168.0.254:5002/
---
cd /media/vadim/1TB_SSD/my_github/meter-watch/services/person-detector
conda activate YOLO_t0
python run.py

## ONE CMD
cd /media/vadim/1TB_SSD/my_github/meter-watch/services/person-detector && conda activate YOLO_t0 && python run.py

http://192.168.0.254:5000/api/status
---

cd /media/vadim/1TB_SSD/my_github/meter-watch/web/meter-watch
ng serve

http://localhost:4200/recognition


----

pgAdmin: http://localhost:5050
Логин: admin@example.com
Пароль: admin_password
При первом входе добавьте сервер: хост postgres, порт 5432







## 📋 **Структурированная организация заметок**

### 1. **Настройка окружения** (Environment Setup)
```bash
# Основные зависимости
pip install sqlalchemy
pip install psycopg2-binary  # или asyncpg

# Миграции БД
pip install alembic
alembic init -t async alembic
alembic revision --autogenerate -m "Initial tables"
alembic upgrade head
```

### 2. **Работа с БД** (Database Management)

docker compose up postgres

**Команды миграций:**
```bash
# Проверка состояния
alembic current
alembic history

# Откаты
alembic downgrade -1
alembic downgrade <revision_id>
alembic downgrade base

# Создание новой миграции
alembic revision --autogenerate -m "Add confidence to meter_readings"
alembic upgrade head

# Пересоздание БД (ОСТОРОЖНО!)
alembic downgrade base && alembic upgrade head
```

**Бекап перед продакшеном:**
```bash
docker exec person_tracker_postgres pg_dump -U tracker_user person_tracker > backup_$(date +%Y%m%d).sql
```

### 3. **Запуск сервисов** (Services)

#### **Docker Compose:**
```bash
# Все сервисы
docker compose up

# Отдельные сервисы
docker compose up postgres
docker compose up redis
docker compose up redis-commander
docker compose up grafana

# С профилями инструментов
docker compose --profile tools up -d
```

#### **Сервисы распознавания:**
| Сервис | Путь | Команда | Порт |
|--------|------|---------|------|
| **CNN Recognizer** | `services/cnn-recognizer` | `conda activate /path/to/.conda && python app.py` | 5002 |
| **Person Detector** | `services/person-detector` | `conda activate YOLO_t0 && python run.py` | 5000 |

**Быстрые команды:**
```bash
# CNN Recognizer
cd /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer && conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda && python app.py

# Person Detector
cd /media/vadim/1TB_SSD/my_github/meter-watch/services/person-detector && conda activate YOLO_t0 && python run.py
```

### 4. **URL-ы и доступы** (URLs & Access)

| Назначение | URL | Примечание |
|------------|-----|------------|
| **Web Admin** | `http://localhost:8080/recognition` | Основной интерфейс |
| **Web Admin (IP)** | `http://192.168.0.254:8080/recognition` | Сеть |
| **Grafana** | `http://localhost:3000/d/meter-watch-dashboard/...` | Дашборды |
| **Redis Commander** | `http://localhost:8081/` | Управление Redis |
| **pgAdmin** | `http://localhost:5050` | Логин: admin@example.com / admin_password |

**API эндпоинты:**
```bash
# Person Detector API
http://192.168.0.254:5000/api/status
http://192.168.0.254:5000/api/set/gas_flow

# Dev
http://192.168.0.254:5858/next_image
```

**setting**
"camera_url": "http://192.168.0.254:5858/next_image",

### 5. **Docker управление** (Docker Management)

#### **Работа с образами:**
```bash
# Просмотр
docker images
docker ps

# Сборка (с звуковым уведомлением)
docker compose build 2>&1 | tee ./build.log && (echo -e "\a"; sleep 0.1; echo -e "\a") && echo "✅ Done!"

# Пересборка конкретного сервиса
docker compose build person-detector
```

#### **Отладка:**
```bash
# Логи
docker logs -f cnn-recognition-app
docker logs --tail 50 cnn-recognition-app

# Вход в контейнер
docker exec -it person-detector-app /bin/bash

# Перезапуск
docker compose restart
```

#### **Чистка:**
```bash
# Пересоздать с нуля
docker compose down --volumes --remove-orphans && docker compose build --no-cache && docker compose up -d

# Удаление образов
docker rmi meter-watch-backend:latest meter-watch-cnn-recognizer:latest meter-watch-person-detector:latest

# Удаление томов
docker volume rm meter-watch_grafana_data
```

### 6. **Структура проекта** (Project Structure)

```bash
# Показать структуру (исключая ненужные папки)
tree -I "data|web|node_modules|__pycache__|logs|output|validation|*.pyc"
```

### 7. **Развертывание на сервере** (Server Deployment)

#### Test Server

ssh root@192.168.0.53
root
12345


```bash
# Установка Docker (Proxmox)
bash -c "$(curl -fsSL https://raw.githubusercontent.com/community-scripts/ProxmoxVE/main/ct/docker.sh)"

# Перенос образов
docker save person-tracker-base:latest | pv | ssh root@192.168.0.53 'docker load'
```



#### not work maybe old docker client
docker save person-tracker-base:latest meter-watch-person-detector:latest meter-watch-cnn-recognizer:latest meter-watch-frontend:latest | pv | ssh root@192.168.0.53 'docker load'


docker save person-tracker-base:latest  | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-cnn-recognizer:latest | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-person-detector:latest | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-frontend:latest | pv | ssh root@192.168.0.53 'docker load'



#### connect

docker exec -it person-detector-app /bin/bash
docker exec -it cnn-recognition-app /bin/bash
docker exec -it person_tracker_redis /bin/bash
docker exec -it redis_commander /bin/bash


#### build with sound notification

docker compose build 2>&1 | tee ./build.log && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


docker compose build && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


### 8. **Полезные сниппеты** (Useful Snippets)

sudo rm -rf /media/vadim/1TB_SSD/my_github/meter-watch/output/wrong_predictions/*



#### **Символические ссылки для импортов:**
```bash
cd /home/vadim/miniconda3/envs/YOLO_t0/lib/python3.10/site-packages/
ln -s /media/vadim/1TB_SSD/my_github/meter-watch/meter-watch-shared meter_watch_shared
```

##### Проверить
python -c "import meter_watch_shared; print('✅ Found')"


#### **Инструменты в контейнерах:**
```bash
apt-get update && apt-get install -y iputils-ping procps
ps aux | grep python
pkill -f "python app.py"
```

#### **TensorBoard:**
```bash
tensorboard --logdir=runs
```

### 9. **Решение проблем** (Troubleshooting)

**Проблемы с импортами:**
```bash
pip uninstall meter-watch-shared -y
python -m pip install -e . --no-user
```

**Если контейнер перезапускается:**
```dockerfile
# Закомментировать CMD в Dockerfile
# CMD ["python", "main.py"]
```


### Clear Grafana Database

Option 4: Clear the Database (Nuclear Option)
WARNING: This will delete ALL Grafana data!

bash
# Find the volume name
docker volume ls | grep grafana

# Remove the volume
docker volume rm <grafana-volume-name>

# Or with docker compose
docker compose down -v
docker compose up -d


docker volume ls | grep grafana
local     meter-watch_grafana_data

docker volume rm meter-watch_grafana_data
---

## 💡 **Рекомендации:**

1. **Используйте .env файл** для переменных окружения
2. **Создайте Makefile** для автоматизации частых команд
3. **Добавьте теги** в заметки (например, #docker, #db, #deploy)
4. **Храните это в отдельном markdown-файле** в корне проекта (`DEVELOPMENT.md` или `NOTES.md`)
