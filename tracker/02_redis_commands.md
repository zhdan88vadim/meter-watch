Excellent question! Here are the **most essential Redis commands** you'll use 90% of the time when building applications. I've organized them by category with practical examples.

---

## 📝 1. Basic String Operations (Most Common)

Strings are the simplest and most used Redis data type.

| Command | What It Does | Example |
|---------|--------------|---------|
| **`SET key value`** | Store a value | `SET user:1 "John"` |
| **`GET key`** | Retrieve a value | `GET user:1` → `"John"` |
| **`DEL key`** | Delete a key | `DEL user:1` |
| **`EXISTS key`** | Check if key exists | `EXISTS user:1` → `1` (true) |
| **`EXPIRE key seconds`** | Set timeout in seconds | `EXPIRE session:abc 3600` |
| **`TTL key`** | Check remaining time | `TTL session:abc` → `3599` |
| **`INCR key`** | Increment counter | `INCR page:views` |
| **`MSET k1 v1 k2 v2`** | Set multiple keys at once | `MSET name "Bob" age "30"` |
| **`MGET k1 k2`** | Get multiple keys | `MGET name age` → `["Bob", "30"]` |

### Python Examples:
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Basic operations
r.set('user:1', 'John')
user = r.get('user:1')  # b'John'

# With expiry
r.setex('session:abc', 3600, 'data')  # Expires in 1 hour
# or
r.set('session:abc', 'data', ex=3600)

# Counter
r.incr('page:views')  # Increments by 1
r.incrby('page:views', 5)  # Increment by 5

# Multiple keys
r.mset({'name': 'Bob', 'age': 30})
results = r.mget(['name', 'age'])  # [b'Bob', b'30']
```

---

## 📂 2. Hashes (For Objects/Structured Data)

Hashes are perfect for storing objects (like user profiles, settings, or your detection data).

| Command | What It Does | Example |
|---------|--------------|---------|
| **`HSET key field value`** | Set field in hash | `HSET user:1 name "John" age 30` |
| **`HGET key field`** | Get a field value | `HGET user:1 name` → `"John"` |
| **`HGETALL key`** | Get all fields | `HGETALL user:1` |
| **`HMGET key field1 field2`** | Get multiple fields | `HMGET user:1 name age` |
| **`HDEL key field`** | Delete field | `HDEL user:1 age` |
| **`HEXISTS key field`** | Check field exists | `HEXISTS user:1 name` → `1` |
| **`HINCRBY key field amount`** | Increment numeric field | `HINCRBY user:1 score 10` |

### Python Examples:
```python
# Store detection data as hash
r.hset('detection:cam1', mapping={
    'person_id': 1,
    'confidence': 0.95,
    'timestamp': time.time(),
    'bbox': '100,200,150,300'
})

# Get all data for a person
person_data = r.hgetall('detection:cam1')
# {b'person_id': b'1', b'confidence': b'0.95', ...}

# Get specific fields
confidence = r.hget('detection:cam1', 'confidence')
```

---

## 📋 3. Lists (For Sequences/Queues)

Lists are good for queues, logs, and recent activity.

| Command | What It Does | Example |
|---------|--------------|---------|
| **`LPUSH key value`** | Add to left (beginning) | `LPUSH logs "error"` |
| **`RPUSH key value`** | Add to right (end) | `RPUSH queue "job1"` |
| **`LPOP key`** | Remove and get left | `LPOP queue` |
| **`RPOP key`** | Remove and get right | `RPOP queue` |
| **`LRANGE key start stop`** | Get range of elements | `LRANGE logs 0 10` |
| **`LLEN key`** | Get length | `LLEN queue` |

### Python Examples:
```python
# Queue of detection events (newest first)
r.lpush('detection:events', json.dumps({'person': 1, 'time': '...'}))
r.lpush('detection:events', json.dumps({'person': 2, 'time': '...'}))

# Get latest 10 events
events = r.lrange('detection:events', 0, 9)
for event in events:
    print(json.loads(event))
```

---

## 🔢 4. Sets (For Unique Items)

Sets store unique values - great for tracking unique visitors, tags, etc.

| Command | What It Does | Example |
|---------|--------------|---------|
| **`SADD key member`** | Add member to set | `SADD visitors "user:1"` |
| **`SREM key member`** | Remove member | `SREM visitors "user:1"` |
| **`SISMEMBER key member`** | Check if member exists | `SISMEMBER visitors "user:1"` → `1` |
| **`SMEMBERS key`** | Get all members | `SMEMBERS visitors` |
| **`SCARD key`** | Get count of members | `SCARD visitors` |

### Python Examples:
```python
# Track which people are currently in frame
r.sadd('current:people', 'person:1', 'person:2', 'person:3')

# When person leaves
r.srem('current:people', 'person:2')

# Check if someone is still there
is_present = r.sismember('current:people', 'person:1')  # True/False

# Get all current people
all_people = r.smembers('current:people')
```

---

## ⏱️ 5. Key Management Commands

| Command | What It Does | Example |
|---------|--------------|---------|
| **`KEYS pattern`** | Find keys by pattern | `KEYS user:*` (⚠️ Slow with many keys) |
| **`SCAN cursor`** | Iterate over keys (safe) | `SCAN 0 MATCH user:*` |
| **`EXISTS key`** | Check if key exists | `EXISTS user:1` |
| **`TYPE key`** | Get key type | `TYPE user:1` → `hash` |
| **`RENAME old new`** | Rename key | `RENAME user:1 user:2` |
| **`FLUSHDB`** | Clear all keys in current DB | ⚠️ Destructive! |
| **`DBSIZE`** | Get number of keys | `DBSIZE` |

### Python Examples:
```python
# Find all detection keys
keys = r.keys('detection:*')  # Use with caution on large DBs

# Better: Use SCAN for large datasets
cursor = 0
keys = []
while True:
    cursor, results = r.scan(cursor, match='detection:*', count=100)
    keys.extend(results)
    if cursor == 0:
        break

# Check key type
key_type = r.type('user:1')  # b'hash' or b'string' etc.
```

---

## 📊 6. Commands for Your YOLO Tracking Scenario

Here's a complete example of how you'd use these commands in your tracking app:

```python
import redis
import json
import time

r = redis.Redis(host='localhost', port=6379, db=0)

# ===== DETECTION SCRIPT (Publisher) =====
def publish_detection(person_id, bbox, confidence):
    # 1. Store current person data in hash
    person_key = f'person:{person_id}'
    r.hset(person_key, mapping={
        'bbox': str(bbox),
        'confidence': confidence,
        'last_seen': time.time(),
        'first_seen': r.hget(person_key, 'first_seen') or time.time()
    })
    
    # 2. Add to set of active people
    r.sadd('active:people', person_id)
    
    # 3. Publish event to channel (for real-time alerts)
    r.publish('detection:channel', json.dumps({
        'person_id': person_id,
        'timestamp': time.time(),
        'bbox': bbox
    }))
    
    # 4. Add to recent activity list (keep last 1000)
    r.lpush('detection:history', json.dumps({
        'person_id': person_id,
        'timestamp': time.time(),
        'action': 'detected'
    }))
    r.ltrim('detection:history', 0, 999)

# ===== MONITOR SCRIPT (Subscriber) =====
def monitor_people():
    # Get active people count
    active_count = r.scard('active:people')
    print(f"Active people: {active_count}")
    
    # Get all active people with their data
    active_ids = r.smembers('active:people')
    for person_id in active_ids:
        data = r.hgetall(f'person:{person_id.decode()}')
        print(f"Person {person_id.decode()}: {data}")
    
    # Check if someone has been missing for > 5 seconds
    current_time = time.time()
    for person_id in active_ids:
        last_seen = float(r.hget(f'person:{person_id.decode()}', 'last_seen'))
        if current_time - last_seen > 5:
            print(f"⚠️ Person {person_id.decode()} missing for 5+ seconds")
            # Remove from active set
            r.srem('active:people', person_id)
    
    # Get recent events
    recent = r.lrange('detection:history', 0, 10)
    print(f"Recent events: {len(recent)}")

# ===== SUBSCRIBE TO REAL-TIME EVENTS =====
def listen_for_detections():
    pubsub = r.pubsub()
    pubsub.subscribe('detection:channel')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            print(f"🔔 Person {data['person_id']} detected at {data['timestamp']}")
            # Trigger your action here

# Run in a separate thread
import threading
threading.Thread(target=listen_for_detections, daemon=True).start()
```

---

## 📋 Quick Reference Cheat Sheet

### Essential Commands You MUST Know:
```bash
# Starting Redis (if not already running)
redis-server

# Connect to Redis CLI
redis-cli

# Most common commands in one place:
SET key value
GET key
DEL key
EXISTS key
EXPIRE key seconds
TTL key

HSET hash field value
HGET hash field
HGETALL hash

LPUSH list value
RPUSH list value
LPOP list
RPOP list
LRANGE list 0 -1

SADD set member
SREM set member
SISMEMBER set member
SMEMBERS set

# Monitoring
INFO memory
INFO stats
MONITOR  # Watch all commands in real-time

# Cleanup
FLUSHDB  # Delete all keys in current DB
FLUSHALL # Delete all keys in ALL DBs
```

---

## 🎯 Pro Tips for Beginners

1. **Start with Strings and Hashes** - These will cover 80% of your use cases
2. **Use `EXPIRE` for temporary data** - Don't forget to set TTL on sessions/cache
3. **Use Hashes for objects** - Much better than string keys like `user:1:name`, `user:1:age`
4. **Avoid `KEYS *` on production** - Use `SCAN` instead for large datasets
5. **Check TTL** - Always verify your expirations work with `TTL key`
6. **Monitor memory** - Use `INFO memory` to track usage

---

## 🚀 Next Steps

Once you're comfortable with these basics, explore:
- **Sorted Sets** (`ZADD`, `ZRANGE`) - For leaderboards and time-based scoring
- **Pub/Sub** - For real-time messaging (already shown above)
- **Transactions** (`MULTI`/`EXEC`) - For atomic operations
- **Lua Scripting** (`EVAL`) - For complex operations

These commands will handle 90% of what you need in your YOLO tracking application! 🎯