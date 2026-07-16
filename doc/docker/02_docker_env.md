Great question! When you have this in your docker-compose.yml:

```yaml
environment:
  - REDIS_HOST=redis
  - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
```

Here's where the data comes from:

## 1. **Hardcoded Values** (like `REDIS_HOST=redis`)
These are **directly set** in the container. No external lookup needed. The value `redis` will be used as-is.

## 2. **Variable Substitution** (like `${TELEGRAM_BOT_TOKEN}`)
Docker Compose will look for this value in the following order:

### Priority Order:
1. **Environment variables** in your shell where you run `docker-compose up`
2. **`.env` file** in the same directory as your docker-compose.yml
3. **Shell environment** of the Docker daemon (rare)

## How Docker Compose reads `${TELEGRAM_BOT_TOKEN}`:

### Method 1: Using `.env` file (Recommended)
Create a `.env` file in the **same directory as your docker-compose.yml**:

```bash
# .env (same directory as docker-compose.yml)
TELEGRAM_BOT_TOKEN=your_actual_token_here
REDIS_HOST=redis  # Optional, you can also override this
```

### Method 2: Export in your shell
```bash
# Export in your terminal before running docker-compose
export TELEGRAM_BOT_TOKEN=your_actual_token_here
docker-compose up
```

### Method 3: Pass when running
```bash
TELEGRAM_BOT_TOKEN=your_actual_token_here docker-compose up
```

## How your Python code reads these values:

Your Python code typically reads environment variables like this:

```python
# In your config.py or similar
import os

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')  # 'redis' will be the value
```

## Visual Example:

```
Your System
    │
    ├── docker-compose.yml
    │   └── environment:
    │       ├── REDIS_HOST=redis  (hardcoded)
    │       └── TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}  (looks up)
    │
    ├── .env file (optional)
    │   └── TELEGRAM_BOT_TOKEN=abc123
    │
    └── Shell variables
        └── TELEGRAM_BOT_TOKEN=xyz789 (takes priority over .env file)

When you run: docker-compose up

Result in container:
    REDIS_HOST = "redis"
    TELEGRAM_BOT_TOKEN = "xyz789"  (from shell, if exported)
    OR = "abc123"  (from .env file, if no shell var)
    OR = ""  (if neither is set)
```

## Testing what's in your container:

```bash
# Check environment variables inside the container
docker exec <container_name> env | grep TELEGRAM
docker exec <container_name> env | grep REDIS

# Or run Python to check
docker exec <container_name> python -c "import os; print(os.getenv('TELEGRAM_BOT_TOKEN'))"
```

## Important Notes:

1. **`.env` file location**: Must be in the **same directory** as your docker-compose.yml, not in your app directory
2. **No quotes** in `.env` file: Use `TELEGRAM_BOT_TOKEN=abc123` not `TELEGRAM_BOT_TOKEN="abc123"`
3. **Shell variables take priority** over `.env` file
4. **For production**, consider using Docker secrets or a secret manager instead

## Quick Fix for Your Current Setup:

1. Create a `.env` file in the directory where your docker-compose.yml is located:
```bash
# .env
TELEGRAM_BOT_TOKEN=your_actual_bot_token
```

2. Make sure your Python code reads it correctly:
```python
# config.py
import os
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")
```

3. Rebuild and restart:
```bash
docker-compose down
docker-compose up -d
```