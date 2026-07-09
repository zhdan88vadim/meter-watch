import redis
import time
from typing import Optional, Any
from app.config import config
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    _instance = None
    _connection = None
    _last_connection_attempt = 0
    _max_retries = 5
    _retry_delay = 2
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_connection(cls):
        """Получает соединение с Redis с автоматическим переподключением"""
        # Если соединение есть, проверяем его
        if cls._connection is not None:
            try:
                cls._connection.ping()
                return cls._connection
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                logger.warning("⚠️ Redis connection lost, reconnecting...")
                cls._connection = None
        
        # Создаем новое соединение с retry
        retries = 0
        while retries < cls._max_retries:
            try:
                cls._connection = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=config.REDIS_DB,
                    password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=config.REDIS_TIMEOUT,
                    socket_timeout=config.REDIS_TIMEOUT,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                cls._connection.ping()
                cls._last_connection_attempt = time.time()
                logger.info("✅ Connected to Redis")
                return cls._connection
            except redis.exceptions.ConnectionError as e:
                retries += 1
                wait_time = cls._retry_delay * retries
                logger.warning(f"⚠️ Redis connection attempt {retries}/{cls._max_retries} failed: {e}")
                if retries < cls._max_retries:
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("❌ Failed to connect to Redis after maximum retries")
                    raise
        
        raise redis.exceptions.ConnectionError("Could not connect to Redis")
    
    @classmethod
    def reconnect(cls):
        """Принудительное переподключение"""
        cls._connection = None
        return cls.get_connection()
    
    @classmethod
    def _execute(cls, func, *args, **kwargs):
        """Выполняет операцию с обработкой ошибок"""
        try:
            conn = cls.get_connection()
            return func(conn, *args, **kwargs)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.warning(f"⚠️ Redis operation failed: {e}")
            cls.reconnect()
            # Пробуем еще раз
            try:
                conn = cls.get_connection()
                return func(conn, *args, **kwargs)
            except Exception as e2:
                logger.error(f"❌ Redis operation failed after reconnect: {e2}")
                raise
        except Exception as e:
            logger.error(f"❌ Redis operation error: {e}")
            raise
    
    @classmethod
    def get_key(cls, key: str) -> Optional[str]:
        try:
            return cls._execute(lambda conn, k: conn.get(k), key)
        except:
            return None
    
    @classmethod
    def set_key(cls, key: str, value: str, expire: Optional[int] = None):
        try:
            if expire:
                cls._execute(lambda conn, k, v, e: conn.setex(k, e, v), key, value, expire)
            else:
                cls._execute(lambda conn, k, v: conn.set(k, v), key, value)
        except:
            pass
    
    @classmethod
    def delete_key(cls, key: str):
        try:
            cls._execute(lambda conn, k: conn.delete(k), key)
        except:
            pass
    
    @classmethod
    def key_exists(cls, key: str) -> bool:
        try:
            return cls._execute(lambda conn, k: conn.exists(k) > 0, key)
        except:
            return False
    
    @classmethod
    def get_timestamp_key(cls, key: str) -> Optional[float]:
        value = cls.get_key(key)
        return float(value) if value else None
    
    @classmethod
    def set_timestamp_key(cls, key: str, expire: Optional[int] = None) -> float:
        current_time = time.time()
        cls.set_key(key, str(current_time), expire)
        return current_time
    
    @classmethod
    def get_time_since(cls, key: str) -> Optional[float]:
        timestamp = cls.get_timestamp_key(key)
        if timestamp is None:
            return None
        return time.time() - timestamp
    
    @classmethod
    def hset(cls, key: str, mapping: dict):
        try:
            cls._execute(lambda conn, k, m: conn.hset(k, mapping=m), key, mapping)
        except:
            pass
    
    @classmethod
    def hgetall(cls, key: str) -> dict:
        try:
            return cls._execute(lambda conn, k: conn.hgetall(k), key)
        except:
            return {}
    
    @classmethod
    def publish(cls, channel: str, message: str):
        try:
            cls._execute(lambda conn, ch, msg: conn.publish(ch, msg), channel, message)
        except:
            pass
    
    @classmethod
    def lpush(cls, key: str, value: str):
        try:
            cls._execute(lambda conn, k, v: conn.lpush(k, v), key, value)
        except:
            pass
    
    @classmethod
    def ltrim(cls, key: str, start: int, end: int):
        try:
            cls._execute(lambda conn, k, s, e: conn.ltrim(k, s, e), key, start, end)
        except:
            pass
    
    @classmethod
    def sadd(cls, key: str, *values):
        try:
            cls._execute(lambda conn, k, *v: conn.sadd(k, *v), key, *values)
        except:
            pass
    
    @classmethod
    def srem(cls, key: str, value: Any):
        try:
            cls._execute(lambda conn, k, v: conn.srem(k, v), key, value)
        except:
            pass
    
    @classmethod
    def get_pubsub(cls):
        """Получает PubSub объект с переподключением"""
        try:
            conn = cls.get_connection()
            return conn.pubsub()
        except:
            return None