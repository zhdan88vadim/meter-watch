**Нет!** Есть несколько способов автоматизировать это. Вот лучшие решения от простого к сложному:

## ✅ Решение 1: Использовать `.env` файлы для разных окружений (самый простой)

Создайте разные `.env` файлы:

```bash
# .env.local (для локальной разработки)
DATABASE_URL=postgresql://tracker_user:pg_secure_password_here@localhost:5432/person_tracker

# .env.docker (для Docker)
DATABASE_URL=postgresql://tracker_user:secure_password@postgres:5432/person_tracker
```

**В Python:**
```python
import os
from dotenv import load_dotenv
import sys

# Определяем окружение автоматически
def detect_environment():
    # Если запущено в Docker - есть специальная переменная
    if os.getenv("DOCKER_ENV") == "true":
        return "docker"
    # Или проверяем наличие файла .dockerenv
    if os.path.exists("/.dockerenv"):
        return "docker"
    # Или проверяем hostname
    if "docker" in os.getenv("HOSTNAME", ""):
        return "docker"
    return "local"

env = detect_environment()

if env == "docker":
    load_dotenv(".env.docker")
else:
    load_dotenv(".env.local")

DATABASE_URL = os.getenv("DATABASE_URL")
```

**В docker-compose.yml:**
```yaml
app:
  environment:
    - DOCKER_ENV=true
    - DATABASE_URL=postgresql://tracker_user:secure_password@postgres:5432/person_tracker
```

---

## ✅ Решение 2: Использовать переменные окружения (рекомендуемый)

**В Python:**
```python
import os

# Получаем хост из переменной окружения, с fallback на localhost
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "tracker_user")
DB_PASS = os.getenv("DB_PASS", "pg_secure_password_here")
DB_NAME = os.getenv("DB_NAME", "person_tracker")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```

**Локально (на хосте):**
```bash
# Ничего не меняем - используем значения по умолчанию
python app.py
```

**В Docker:**
```yaml
app:
  environment:
    - DB_HOST=postgres  # Только это меняется!
    - DB_USER=tracker_user
    - DB_PASS=secure_password
    - DB_NAME=person_tracker
```

**Преимущество:** Меняете только `DB_HOST`!

---

## ✅ Решение 3: Автоматическое определение через hostname

```python
import os
import socket

def get_db_host():
    """Автоматически определяет хост для БД"""
    hostname = socket.gethostname()
    
    # Если запущено в Docker - hostname обычно длинный
    if "docker" in hostname or os.path.exists("/.dockerenv"):
        return "postgres"  # Имя сервиса в docker-compose
    
    return "localhost"

DATABASE_URL = f"postgresql://tracker_user:pg_secure_password_here@{get_db_host()}:5432/person_tracker"
```

---

## ✅ Решение 4: Использовать `docker-compose.override.yml` (для разработки)

**docker-compose.yml** (общий):
```yaml
services:
  app:
    image: myapp
    environment:
      - DATABASE_URL=postgresql://tracker_user:secure_password@postgres:5432/person_tracker
    # ... остальное
```

**docker-compose.override.yml** (для локальной разработки):
```yaml
services:
  app:
    environment:
      - DATABASE_URL=postgresql://tracker_user:pg_secure_password_here@localhost:5432/person_tracker
```

**Запуск:**
```bash
# Для локальной разработки (использует override)
docker-compose up

# Для продакшна (без override)
docker-compose -f docker-compose.yml up
```

---

## ✅ Решение 5: Использовать переменную окружения с проверкой (мой любимый)

```python
import os

def get_database_url():
    """Умное определение DATABASE_URL"""
    # 1. Если явно задано - используем
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    
    # 2. Проверяем отдельные переменные
    db_host = os.getenv("DB_HOST")
    if db_host:
        db_user = os.getenv("DB_USER", "tracker_user")
        db_pass = os.getenv("DB_PASS", "pg_secure_password_here")
        db_name = os.getenv("DB_NAME", "person_tracker")
        db_port = os.getenv("DB_PORT", "5432")
        return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    # 3. Автоопределение
    if os.path.exists("/.dockerenv"):
        # В Docker
        return "postgresql://tracker_user:secure_password@postgres:5432/person_tracker"
    
    # 4. Локально (по умолчанию)
    return "postgresql://tracker_user:pg_secure_password_here@localhost:5432/person_tracker"

DATABASE_URL = get_database_url()
```

---

## ✅ Решение 6: Через `Makefile` (для разработчиков)

```makefile
# Makefile
.PHONY: local docker

local:
	@echo "Запуск в локальном режиме..."
	@export DATABASE_URL="postgresql://tracker_user:pg_secure_password_here@localhost:5432/person_tracker" && python app.py

docker:
	@echo "Запуск в Docker режиме..."
	@export DATABASE_URL="postgresql://tracker_user:secure_password@postgres:5432/person_tracker" && docker-compose up
```

**Использование:**
```bash
make local   # локальный режим
make docker  # Docker режим
```

---

## ✅ Решение 7: Конфигурация через классы (чистый код)

```python
# config.py
import os
from typing import Optional

class Config:
    """Базовый конфиг"""
    DB_USER = "tracker_user"
    DB_PASS = "pg_secure_password_here"
    DB_NAME = "person_tracker"
    DB_PORT = "5432"
    
    @property
    def DATABASE_URL(self) -> str:
        raise NotImplementedError

class LocalConfig(Config):
    """Конфиг для локальной разработки"""
    DB_HOST = "localhost"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

class DockerConfig(Config):
    """Конфиг для Docker"""
    DB_HOST = "postgres"
    DB_PASS = "secure_password"  # Другой пароль в Docker
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

def get_config():
    """Автоматический выбор конфига"""
    if os.path.exists("/.dockerenv"):
        return DockerConfig()
    return LocalConfig()

config = get_config()
DATABASE_URL = config.DATABASE_URL
```

---

## 🏆 Рекомендация (лучшее решение):

**Используйте переменные окружения + .env файлы:**

```python
# config.py
import os
from dotenv import load_dotenv

# Загружаем .env только если нет переменных окружения
if not os.getenv("DATABASE_URL"):
    env_file = ".env.docker" if os.path.exists("/.dockerenv") else ".env.local"
    load_dotenv(env_file)

DATABASE_URL = os.getenv("DATABASE_URL")
```

**.env.local:**
```bash
DATABASE_URL=postgresql://tracker_user:pg_secure_password_here@localhost:5432/person_tracker
```

**.env.docker:**
```bash
DATABASE_URL=postgresql://tracker_user:secure_password@postgres:5432/person_tracker
```

**В docker-compose.yml:**
```yaml
app:
  environment:
    - DATABASE_URL=postgresql://tracker_user:secure_password@postgres:5432/person_tracker
```

---

## 📊 Итог: что использовать?

| Ситуация | Решение |
|----------|---------|
| **Нужно быстро** | Решение 2 (переменные окружения) |
| **Проект с командой** | Решение 1 (разные .env файлы) |
| **Продакшн** | Решение 5 (умное определение) |
| **Микросервисы** | Решение 7 (конфиг классы) |

**Самое простое для вас:**

```python
# Просто замените в коде:
DB_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_URL = f"postgresql://tracker_user:pg_secure_password_here@{DB_HOST}:5432/person_tracker"
```

**И в Docker добавьте:**
```yaml
environment:
  - DB_HOST=postgres
```

**Вот и всё!** Больше никаких ручных изменений 🎉