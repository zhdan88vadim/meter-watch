Использование Alembic в связке с Docker Compose — правильный путь для управления схемой базы данных. Главное правило: **не используйте инициализационные SQL-скрипты (`/docker-entrypoint-initdb.d`) для создания схемы, если вы уже используете Alembic.** Это приводит к рассинхронизации и проблемам при обновлениях .

Вместо этого нужно, чтобы сама база данных создавалась через `docker-compose.yml`, а миграции применялись либо через `entrypoint` вашего приложения, либо отдельной командой.

### 🔧 Шаг 1: Настройка PostgreSQL в `docker-compose.yml`

Конфигурация базы данных с использованием `POSTGRES_USER`, `POSTGRES_PASSWORD` и `POSTGRES_DB` создаст базу данных автоматически при первом запуске контейнера . Пример `docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16
    container_name: my_postgres_db
    environment:
      POSTGRES_USER: myuser          # Переменная для пользователя
      POSTGRES_PASSWORD: mypassword  # Переменная для пароля
      POSTGRES_DB: mydatabase        # Имя базы данных, которая будет создана
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myuser -d mydatabase"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```

Запустите контейнер: `docker compose up -d db` .

### 🗺️ Шаг 2: Настройка Alembic для работы внутри контейнера

Вашему приложению и Alembic нужно знать, как подключиться к базе.

1.  **Настройка `alembic.ini`**:
    Вместо жесткого прописывания URL, лучше считывать его из переменных окружения в `env.py`. Это стандартная практика для Docker . В `alembic.ini` оставьте строку-заглушку:
    ```ini
    sqlalchemy.url = driver://user:pass@localhost/dbname
    ```

2.  **Настройка `alembic/env.py`**:
    Добавьте код, который динамически формирует URL подключения, забирая данные из переменных окружения, определенных в `docker-compose.yml` или `.env` файле .
    ```python
    # в env.py
    import os
    from logging.config import fileConfig
    from sqlalchemy import engine_from_config, pool
    from alembic import context

    # ... (остальной код)
    config = context.config

    # Получаем данные из переменных окружения
    db_user = os.getenv("POSTGRES_USER", "myuser")
    db_pass = os.getenv("POSTGRES_PASSWORD", "mypassword")
    db_host = os.getenv("POSTGRES_HOST", "db") # "db" — имя сервиса в docker-compose.yml
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "mydatabase")

    # Формируем URL и переопределяем настройку
    config.set_main_option(
        "sqlalchemy.url",
        f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    # ... (остальной код)
    ```

### 🚀 Шаг 3: Применение миграций при запуске

Самый надежный способ убедиться, что схема базы данных всегда актуальна — выполнять `alembic upgrade head` **при старте контейнера** с вашим приложением .

Создайте `entrypoint.sh`:

```bash
#!/bin/bash
set -e

echo "Ожидание готовности базы данных..."
# Можно добавить цикл ожидания с помощью, например, nc или dockerize

echo "Применение миграций Alembic..."
alembic upgrade head

echo "Запуск приложения..."
exec "$@"
```

И добавьте его в ваш `Dockerfile`:

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# CMD для вашего приложения
CMD ["python", "app.py"]
```

### 📝 Дополнительные команды

*   **Создание новой миграции**: `docker compose exec <app_service_name> alembic revision --autogenerate -m "описание"` .
*   **Применение вручную**: `docker compose exec <app_service_name> alembic upgrade head` .
*   **Проверка состояния**: `docker compose exec <app_service_name> alembic current` .
*   **Полный сброс и повторное применение**: `docker compose down -v && docker compose up -d` (осторожно, удалит все данные) .

Эта схема гарантирует, что ваша база данных будет создана Docker-ом, а схема в ней всегда будет соответствовать коду вашего приложения благодаря Alembic, запускаемому при каждом старте контейнера.