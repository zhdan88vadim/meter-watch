import redis
import time
import threading
from typing import Optional, Any, Dict, List, Union
from redis.connection import ConnectionPool
from app.config import config
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    """Менеджер для работы с Redis с пулом соединений и автоматическим восстановлением"""
    
    _instance = None
    _pool: Optional[ConnectionPool] = None
    _lock = threading.Lock()
    _max_retries = 3
    _retry_delay = 1
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _get_pool(cls) -> ConnectionPool:
        """Создает или возвращает существующий пул соединений"""
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    try:
                        cls._pool = ConnectionPool(
                            host=config.REDIS_HOST,
                            port=config.REDIS_PORT,
                            db=config.REDIS_DB,
                            password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
                            decode_responses=True,
                            socket_connect_timeout=config.REDIS_TIMEOUT,
                            socket_timeout=config.REDIS_TIMEOUT,
                            retry_on_timeout=True,
                            health_check_interval=30,
                            max_connections=20,
                            retry_on_error=[redis.exceptions.ConnectionError, 
                                          redis.exceptions.TimeoutError]
                        )
                        logger.info("✅ Redis connection pool created")
                    except Exception as e:
                        logger.error(f"❌ Failed to create Redis connection pool: {e}")
                        raise
        return cls._pool
    
    @classmethod
    def get_connection(cls) -> redis.Redis:
        """Получает клиент Redis из пула"""
        try:
            pool = cls._get_pool()
            client = redis.Redis(connection_pool=pool)
            # Проверяем соединение
            client.ping()
            return client
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.warning(f"⚠️ Redis connection error: {e}")
            # Сбрасываем пул чтобы создать новый
            with cls._lock:
                if cls._pool:
                    cls._pool.disconnect()
                    cls._pool = None
            raise
    
    @classmethod
    def execute(cls, operation: str, func, *args, **kwargs):
        """Выполняет операцию с автоматическим восстановлением при ошибках"""
        last_error = None
        
        for attempt in range(cls._max_retries):
            try:
                client = cls.get_connection()
                result = func(client, *args, **kwargs)
                return result
                
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                last_error = e
                wait_time = cls._retry_delay * (attempt + 1)
                logger.warning(
                    f"⚠️ Redis {operation} failed (attempt {attempt + 1}/{cls._max_retries}): {e}"
                )
                
                if attempt < cls._max_retries - 1:
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    # Сбрасываем соединение для следующей попытки
                    with cls._lock:
                        if cls._pool:
                            cls._pool.disconnect()
                            cls._pool = None
                else:
                    logger.error(f"❌ Redis {operation} failed after {cls._max_retries} attempts")
                    
            except redis.exceptions.RedisError as e:
                logger.error(f"❌ Redis {operation} error: {e}")
                raise
            except Exception as e:
                logger.error(f"❌ Unexpected error in Redis {operation}: {e}")
                raise
        
        if last_error:
            raise last_error
        raise redis.exceptions.ConnectionError(f"Could not execute {operation}")
    
    # === Базовые операции ===
    
    @classmethod
    def get_key(cls, key: str) -> Optional[str]:
        """Получает значение по ключу"""
        try:
            return cls.execute("get", lambda c, k: c.get(k), key)
        except Exception as e:
            logger.error(f"Failed to get key '{key}': {e}")
            return None
    
    @classmethod
    def set_key(cls, key: str, value: Union[str, bytes], expire: Optional[int] = None) -> bool:
        """Устанавливает значение с опциональным TTL"""
        try:
            if expire:
                cls.execute("setex", lambda c, k, v, e: c.setex(k, e, v), key, value, expire)
            else:
                cls.execute("set", lambda c, k, v: c.set(k, v), key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set key '{key}': {e}")
            return False
    
    @classmethod
    def delete_key(cls, key: str) -> bool:
        """Удаляет ключ"""
        try:
            result = cls.execute("delete", lambda c, k: c.delete(k), key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            return False
    
    @classmethod
    def key_exists(cls, key: str) -> bool:
        """Проверяет существование ключа"""
        try:
            return bool(cls.execute("exists", lambda c, k: c.exists(k), key))
        except Exception as e:
            logger.error(f"Failed to check key '{key}': {e}")
            return False
    
    @classmethod
    def expire(cls, key: str, seconds: int) -> bool:
        """Устанавливает TTL для ключа"""
        try:
            return bool(cls.execute("expire", lambda c, k, s: c.expire(k, s), key, seconds))
        except Exception as e:
            logger.error(f"Failed to set expire for key '{key}': {e}")
            return False
    
    @classmethod
    def ttl(cls, key: str) -> int:
        """Возвращает оставшееся время жизни ключа в секундах"""
        try:
            return cls.execute("ttl", lambda c, k: c.ttl(k), key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key '{key}': {e}")
            return -2  # -2 значит ключ не существует
    
    # === Timestamp операции ===
    
    @classmethod
    def get_timestamp_key(cls, key: str) -> Optional[float]:
        """Получает timestamp из ключа"""
        try:
            value = cls.get_key(key)
            if value:
                return float(value)
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse timestamp for key '{key}': {e}")
            return None
    
    @classmethod
    def set_timestamp_key(cls, key: str, expire: Optional[int] = None) -> float:
        """Устанавливает текущий timestamp в ключ"""
        current_time = time.time()
        cls.set_key(key, str(current_time), expire)
        return current_time
    
    @classmethod
    def get_time_since(cls, key: str) -> Optional[float]:
        """Возвращает время в секундах с момента установки timestamp"""
        timestamp = cls.get_timestamp_key(key)
        if timestamp is None:
            return None
        return time.time() - timestamp
    
    # === Hash операции ===
    
    @classmethod
    def hset(cls, key: str, mapping: Dict[str, Any]) -> bool:
        """Устанавливает поля в hash"""
        try:
            # Преобразуем значения в строки для совместимости
            str_mapping = {k: str(v) if v is not None else None 
                          for k, v in mapping.items()}
            cls.execute("hset", lambda c, k, m: c.hset(k, mapping=m), key, str_mapping)
            return True
        except Exception as e:
            logger.error(f"Failed to hset key '{key}': {e}")
            return False
    
    @classmethod
    def hgetall(cls, key: str) -> Dict[str, str]:
        """Получает все поля из hash"""
        try:
            return cls.execute("hgetall", lambda c, k: c.hgetall(k), key) or {}
        except Exception as e:
            logger.error(f"Failed to hgetall key '{key}': {e}")
            return {}
    
    @classmethod
    def hget(cls, key: str, field: str) -> Optional[str]:
        """Получает значение поля из hash"""
        try:
            return cls.execute("hget", lambda c, k, f: c.hget(k, f), key, field)
        except Exception as e:
            logger.error(f"Failed to hget key '{key}' field '{field}': {e}")
            return None
    
    @classmethod
    def hdel(cls, key: str, *fields) -> bool:
        """Удаляет поля из hash"""
        try:
            result = cls.execute("hdel", lambda c, k, *f: c.hdel(k, *f), key, *fields)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to hdel key '{key}': {e}")
            return False
    
    # === List операции ===
    
    @classmethod
    def lpush(cls, key: str, *values) -> Optional[int]:
        """Добавляет значения в начало списка"""
        try:
            return cls.execute("lpush", lambda c, k, *v: c.lpush(k, *v), key, *values)
        except Exception as e:
            logger.error(f"Failed to lpush key '{key}': {e}")
            return None
    
    @classmethod
    def rpush(cls, key: str, *values) -> Optional[int]:
        """Добавляет значения в конец списка"""
        try:
            return cls.execute("rpush", lambda c, k, *v: c.rpush(k, *v), key, *values)
        except Exception as e:
            logger.error(f"Failed to rpush key '{key}': {e}")
            return None
    
    @classmethod
    def lpop(cls, key: str) -> Optional[str]:
        """Удаляет и возвращает первый элемент списка"""
        try:
            return cls.execute("lpop", lambda c, k: c.lpop(k), key)
        except Exception as e:
            logger.error(f"Failed to lpop key '{key}': {e}")
            return None
    
    @classmethod
    def lrange(cls, key: str, start: int, end: int) -> List[str]:
        """Получает элементы списка в диапазоне"""
        try:
            return cls.execute("lrange", lambda c, k, s, e: c.lrange(k, s, e), key, start, end) or []
        except Exception as e:
            logger.error(f"Failed to lrange key '{key}': {e}")
            return []
    
    @classmethod
    def ltrim(cls, key: str, start: int, end: int) -> bool:
        """Обрезает список до указанного диапазона"""
        try:
            cls.execute("ltrim", lambda c, k, s, e: c.ltrim(k, s, e), key, start, end)
            return True
        except Exception as e:
            logger.error(f"Failed to ltrim key '{key}': {e}")
            return False
    
    # === Set операции ===
    
    @classmethod
    def sadd(cls, key: str, *values) -> Optional[int]:
        """Добавляет элементы в множество"""
        try:
            return cls.execute("sadd", lambda c, k, *v: c.sadd(k, *v), key, *values)
        except Exception as e:
            logger.error(f"Failed to sadd key '{key}': {e}")
            return None
    
    @classmethod
    def srem(cls, key: str, *values) -> Optional[int]:
        """Удаляет элементы из множества"""
        try:
            return cls.execute("srem", lambda c, k, *v: c.srem(k, *v), key, *values)
        except Exception as e:
            logger.error(f"Failed to srem key '{key}': {e}")
            return None
    
    @classmethod
    def smembers(cls, key: str) -> set:
        """Получает все элементы множества"""
        try:
            return cls.execute("smembers", lambda c, k: c.smembers(k), key) or set()
        except Exception as e:
            logger.error(f"Failed to smembers key '{key}': {e}")
            return set()
    
    @classmethod
    def sismember(cls, key: str, value: str) -> bool:
        """Проверяет наличие элемента в множестве"""
        try:
            return bool(cls.execute("sismember", lambda c, k, v: c.sismember(k, v), key, value))
        except Exception as e:
            logger.error(f"Failed to sismember key '{key}': {e}")
            return False
    
    # === Pub/Sub операции ===
    
    @classmethod
    def publish(cls, channel: str, message: str) -> Optional[int]:
        """Публикует сообщение в канал"""
        try:
            return cls.execute("publish", lambda c, ch, msg: c.publish(ch, msg), channel, message)
        except Exception as e:
            logger.error(f"Failed to publish to channel '{channel}': {e}")
            return None
    
    @classmethod
    def get_pubsub(cls) -> Optional[redis.client.PubSub]:
        """Создает PubSub объект"""
        try:
            client = cls.get_connection()
            return client.pubsub()
        except Exception as e:
            logger.error(f"Failed to create pubsub: {e}")
            return None
    
    # === Pipeline операция ===
    
    @classmethod
    def pipeline(cls, transaction: bool = True) -> Optional[redis.client.Pipeline]:
        """Создает pipeline для групповых операций"""
        try:
            client = cls.get_connection()
            return client.pipeline(transaction=transaction)
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return None
    
    # === Утилиты ===
    
    @classmethod
    def reconnect(cls):
        """Принудительное переподключение (сброс пула)"""
        with cls._lock:
            if cls._pool:
                try:
                    cls._pool.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting pool: {e}")
                cls._pool = None
        logger.info("🔄 Redis connection pool reset")
    
    @classmethod
    def close(cls):
        """Закрывает все соединения"""
        with cls._lock:
            if cls._pool:
                try:
                    cls._pool.disconnect()
                    logger.info("✅ Redis connection pool closed")
                except Exception as e:
                    logger.error(f"Error closing Redis pool: {e}")
                cls._pool = None
            cls._instance = None