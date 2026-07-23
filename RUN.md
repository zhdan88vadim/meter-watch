pip install sqlalchemy

pip install psycopg2-binary
OR
pip install asyncpg


pip install alembic
alembic init -t async alembic

conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda
alembic revision --autogenerate -m "Initial tables"

alembic upgrade head

#### update db
alembic revision --autogenerate -m "Add confidence to meter_readings"
alembic upgrade head


<!-- 

Всегда делайте бекап перед миграцией в продакшене:

bash
docker exec person_tracker_postgres pg_dump -U tracker_user person_tracker > backup_$(date +%Y%m%d).sql


# Проверить текущую версию базы
alembic current

# Посмотреть историю миграций
alembic history

# Откатить на одну миграцию назад
alembic downgrade -1

# Откатить к конкретной ревизии
alembic downgrade <revision_id>

# Откатить все миграции
alembic downgrade base

# Пересоздать базу с нуля (ОСТОРОЖНО!)
alembic downgrade base && alembic upgrade head

# Создать пустую миграцию вручную
alembic revision -m "Manual migration" -->




## URLS


### grafana

http://localhost:3000/d/meter-watch-dashboard/meter-watch-dashboard?orgId=1&from=now-24h&to=now&timezone=browser&var-source_filter=&var-event_type_filter=&refresh=30s


### admin web - docker

http://localhost:8080/recognition
http://192.168.0.254:8080/recognition


## person detector api

docker
http://192.168.0.254:5000/status





## dev run

docker compose up redis-commander
http://localhost:8081/

---

cd /media/vadim/1TB_SSD/my_github/meter-watch/cnn/backend
conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda
python app.py

## ONE CMD
cd /media/vadim/1TB_SSD/my_github/meter-watch/cnn/backend && conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda && python app.py



http://192.168.0.254:5002/
---
cd /media/vadim/1TB_SSD/my_github/meter-watch/person-detector
conda activate YOLO_t0
python run.py

## ONE CMD
cd /media/vadim/1TB_SSD/my_github/meter-watch/person-detector && conda activate YOLO_t0 && python run.py

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


test
nc -zv localhost 5432


### bash

sudo rm -rf /media/vadim/1TB_SSD/my_github/meter-watch/output/wrong_predictions/*

==============================================





conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda
conda activate YOLO_t0


tensorboard --logdir=runs

docker compose --profile tools up -d


docker ps


# Посмотреть логи контейнера
docker logs redis_commander

# Или последние логи
docker logs --tail 50 redis_commander


WEB UI
http://localhost:8080/recognition


Redis Commander
http://localhost:8081/


http://192.168.0.254:5000/api/status

http://192.168.0.254:5000/api/set/gas_flow







docker compose up


# need rebuild after update Dockerfile
docker compose down

docker compose build

# how to debug if loop reboot

need to comment 
<!-- 
# Команда для запуска
# CMD ["python", "main.py"] -->


# connect

docker exec -it person-detector-app /bin/bash
docker exec -it cnn-recognition-app /bin/bash
docker exec -it person_tracker_redis /bin/bash
docker exec -it redis_commander /bin/bash



docker compose build 2>&1 | tee ./build.log && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


docker compose build && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


docker compose up --build


# build

docker compose build person-detector


# run docker

docker compose up person-detector-app
docker compose up cnn-recognition-app

docker compose up redis
docker compose up redis-commander


docker compose restart
Эта команда просто останавливает и снова запускает уже существующие контейнеры.



# run only redis
docker compose up redis





apt-get update && apt-get install -y iputils-ping
ping -c 3 192.168.0.102




# logs
docker ps -a

docker logs -f cnn-recognition-app

docker logs --tail 50 cnn-recognition-app

docker logs -f -t --tail 100 cnn-recognition-app





Решение 1: Создать символическую ссылку (Самый надежный)
bash
# Создать символическую ссылку в conda site-packages
cd /home/vadim/miniconda3/envs/YOLO_t0/lib/python3.10/site-packages/
ln -s /media/vadim/1TB_SSD/my_github/meter-watch/meter-watch-shared meter_watch_shared

# Проверить
python -c "import meter_watch_shared; print('✅ Found')"



pip uninstall meter-watch-shared -y

python -m pip install -e . --no-user



# tools


apt-get update && apt-get install -y procps

ps aux | grep python

pkill -f "python app.py"





Откройте браузер: http://localhost:5050
Логин:
Email: admin@example.com
Password: admin_password
При добавлении сервера:
Нажмите "Add New Server"
Вкладка "General":
Name: Person Tracker DB (или любое имя)
Вкладка "Connection" (самое важное!):
Host name/address: postgres ⬅️ НЕ localhost!
Port: 5432
Maintenance database: person_tracker
Username: tracker_user
Password: secure_password
✅ "Save password?"




## docker analyze

vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/meter-watch$ docker compose images
WARN[0000] /media/vadim/1TB_SSD/my_github/meter-watch/docker-compose.yml: `version` is obsolete 
CONTAINER                 REPOSITORY                       TAG                 IMAGE ID            SIZE
angular-frontend          meter-watch-frontend             latest              2d7ff41d62b1        64.5MB
cnn-recognition-app       meter-watch-cnn-recognizer       latest              6bc47538c3e3        7.65GB
person-detector-app       meter-watch-person-detector      latest              1c2e302c8b4d        6.01GB
person_tracker_grafana    grafana/grafana                  latest              7ddecc39af95        1.16GB
person_tracker_pgadmin    dpage/pgadmin4                   latest              a389978bd3d4        512MB
person_tracker_postgres   postgres                         16-alpine           de3a4eab8fdf        294MB
person_tracker_redis      redis                            7.2-alpine          b6636bae9624        38.6MB
redis_commander           rediscommander/redis-commander   latest              778af9bd6397        77.8MB




vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/meter-watch$ 

docker history meter-watch-cnn-recognizer:latest --human

IMAGE          CREATED             CREATED BY                                      SIZE      COMMENT
6bc47538c3e3   About an hour ago   CMD ["python" "app.py"]                         0B        buildkit.dockerfile.v0
<missing>      About an hour ago   RUN /bin/sh -c pip install -e /app/meter_wat…   4.62MB    buildkit.dockerfile.v0
<missing>      About an hour ago   COPY meter-watch-shared /app/meter_watch_sha…   120kB     buildkit.dockerfile.v0
<missing>      2 hours ago         COPY cnn/backend /app # buildkit                45.8MB    buildkit.dockerfile.v0
<missing>      2 hours ago         RUN /bin/sh -c pip install -r requirements.t…   7.17GB    buildkit.dockerfile.v0
<missing>      2 hours ago         COPY cnn/backend/requirements.txt . # buildk…   247B      buildkit.dockerfile.v0
<missing>      3 months ago        RUN /bin/sh -c apt-get update && apt-get ins…   304MB     buildkit.dockerfile.v0
<missing>      3 months ago        WORKDIR /app                                    0B        buildkit.dockerfile.v0
<missing>      8 months ago        CMD ["python3"]                                 0B        buildkit.dockerfile.v0
<missing>      8 months ago        RUN /bin/sh -c set -eux;  for src in idle3 p…   36B       buildkit.dockerfile.v0
<missing>      8 months ago        RUN /bin/sh -c set -eux;   savedAptMark="$(a…   39.5MB    buildkit.dockerfile.v0
<missing>      8 months ago        ENV PYTHON_SHA256=00e07d7c0f2f0cc002432d1ee8…   0B        buildkit.dockerfile.v0
<missing>      8 months ago        ENV PYTHON_VERSION=3.9.25                       0B        buildkit.dockerfile.v0
<missing>      8 months ago        ENV GPG_KEY=E3FF2839C048B25C084DEBE9B26995E3…   0B        buildkit.dockerfile.v0
<missing>      8 months ago        RUN /bin/sh -c set -eux;  apt-get update;  a…   3.81MB    buildkit.dockerfile.v0
<missing>      8 months ago        ENV LANG=C.UTF-8                                0B        buildkit.dockerfile.v0
<missing>      8 months ago        ENV PATH=/usr/local/bin:/usr/local/sbin:/usr…   0B        buildkit.dockerfile.v0
<missing>      9 months ago        # debian.sh --arch 'amd64' out/ 'trixie' '@1…   78.6MB    debuerreotype 0.16






docker history meter-watch-person-detector:latest --human


vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/meter-watch$ docker history meter-watch-person-detector:latest --human
IMAGE          CREATED             CREATED BY                                      SIZE      COMMENT
1c2e302c8b4d   About an hour ago   CMD ["python" "run.py"]                         0B        buildkit.dockerfile.v0
<missing>      About an hour ago   RUN /bin/sh -c pip install -e /app/meter_wat…   1.6MB     buildkit.dockerfile.v0
<missing>      About an hour ago   COPY meter-watch-shared /app/meter_watch_sha…   120kB     buildkit.dockerfile.v0
<missing>      About an hour ago   COPY person-detector /app # buildkit            14.3MB    buildkit.dockerfile.v0
<missing>      3 hours ago         RUN /bin/sh -c apt-get update && apt-get ins…   127MB     buildkit.dockerfile.v0
<missing>      3 hours ago         RUN /bin/sh -c pip install -r requirements.t…   5.43GB    buildkit.dockerfile.v0
<missing>      3 hours ago         COPY person-detector/requirements.txt . # bu…   229B      buildkit.dockerfile.v0
<missing>      7 days ago          RUN /bin/sh -c apt-get update && apt-get ins…   309MB     buildkit.dockerfile.v0
<missing>      7 days ago          WORKDIR /app                                    0B        buildkit.dockerfile.v0
<missing>      7 days ago          CMD ["python3"]                                 0B        buildkit.dockerfile.v0
<missing>      7 days ago          RUN /bin/sh -c set -eux;  for src in idle3 p…   36B       buildkit.dockerfile.v0
<missing>      7 days ago          RUN /bin/sh -c set -eux;   savedAptMark="$(a…   39.7MB    buildkit.dockerfile.v0
<missing>      7 days ago          ENV PYTHON_SHA256=de6517421601e39a9a3bc3e1bc…   0B        buildkit.dockerfile.v0
<missing>      7 days ago          ENV PYTHON_VERSION=3.10.20                      0B        buildkit.dockerfile.v0
<missing>      7 days ago          ENV GPG_KEY=A035C8C19219BA821ECEA86B64E628F8…   0B        buildkit.dockerfile.v0
<missing>      7 days ago          RUN /bin/sh -c set -eux;  apt-get update;  a…   3.81MB    buildkit.dockerfile.v0
<missing>      7 days ago          ENV LANG=C.UTF-8                                0B        buildkit.dockerfile.v0
<missing>      7 days ago          ENV PATH=/usr/local/bin:/usr/local/sbin:/usr…   0B        buildkit.dockerfile.v0
<missing>      8 days ago          # debian.sh --arch 'amd64' out/ 'trixie' '@1…   78.6MB    debuerreotype 0.17






The ultralytics==8.4.95 package itself does not hardcode a single specific PyTorch version, but it does have a minimum requirement. Based on the project's official documentation, ultralytics requires PyTorch>=1.7.

However, for the best performance and compatibility with the latest features, the Ultralytics team recommends installing a more recent version of PyTorch (for example, torch>=1.8). The exact version that pip installs will be the latest stable one compatible with your system and CUDA setup, not a fixed one tied to the ultralytics version.

Because PyTorch requirements can vary depending on your operating system and whether you want GPU support, the official guide suggests installing PyTorch first from the official website (pytorch.org) before you install ultralytics. This ensures you get the correct build for your hardware.




# Build base image
docker build -t person-tracker-base:latest -f docker_base/Dockerfile .


docker compose build person-detector
docker compose build cnn-recognizer



vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/meter-watch$ docker system df -v | grep -A 20 "Images"
Images space usage:

REPOSITORY                       TAG          IMAGE ID       CREATED          SIZE      SHARED SIZE   UNIQUE SIZE   CONTAINERS
meter-watch-person-detector      latest       e25770764592   9 minutes ago    6.19GB    5.818GB       376.9MB       0
meter-watch-cnn-recognizer       latest       7e161589b00f   9 minutes ago    5.92GB    5.818GB       101.2MB       0
person-tracker-base              latest       df5ec35219a1   12 minutes ago   5.82GB    5.818GB       0B            0
<none>                           <none>       fe9d3b3b41eb   5 hours ago      2.04GB    2.021GB       16.07MB       1
<none>                           <none>       ae8038ee69fe   19 hours ago     2.04GB    2.021GB       16.07MB       0
<none>                           <none>       b278cdb80312   19 hours ago     1.62GB    425.7MB       1.197GB       1
<none>                           <none>       6119bd1b1015   19 hours ago     2.04GB    2.021GB       16.07MB       0
<none>                           <none>       6bc47538c3e3   21 hours ago     7.65GB    7.641GB       4.743MB       0
<none>                           <none>       1c2e302c8b4d   21 hours ago     6.01GB    5.993GB       16.07MB       0
<none>                           <none>       44e69c146c06   22 hours ago     7.65GB    7.641GB       4.742MB       0
<none>                           <none>       5964737cc8b8   22 hours ago     6.01GB    5.993GB       16.07MB       0
meter-watch-frontend             latest       2d7ff41d62b1   23 hours ago     64.5MB    62.36MB       2.094MB       1
<none>                           <none>       4d8f1a93db01   23 hours ago     7.61GB    7.562GB       50.5MB        0
<none>                           <none>       917f4ee2bb1c   23 hours ago     5.97GB    5.957GB       16.07MB       0
grafana/grafana                  latest       7ddecc39af95   43 hours ago     1.16GB    8.416MB       1.149GB       1
<none>                           <none>       ce3a1a6f2b09   6 days ago       6.02GB    5.957GB       63.48MB       0
<none>                           <none>       8ae123fc6a0c   6 days ago       7.59GB    7.589GB       4.733MB       0
<none>                           <none>       2186555e654c   6 days ago       6.02GB    5.955GB       63.48MB       0



## remove images
docker rmi meter-watch-backend:latest meter-watch-cnn-recognizer:latest meter-watch-person-detector:latest

docker rmi meter-watch-cnn-recognizer:latest





vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/meter-watch$ docker images
                                                                                                                  i Info →   U  In Use
IMAGE                                   ID             DISK USAGE   CONTENT SIZE   EXTRA
comp-vision-table-parser:latest         cbc4211661ac       8.23GB             0B        
dpage/pgadmin4:latest                   a389978bd3d4        512MB             0B        
grafana/grafana:latest                  7ddecc39af95       1.16GB             0B        
meter-watch-cnn-recognizer:latest       f00f09f20082        1.6GB             0B        
meter-watch-frontend:latest             2d7ff41d62b1       64.5MB             0B        
meter-watch-person-detector:latest      73a6887e462f       2.25GB             0B        
ollama/ollama:latest                    333628ba5b2f       6.55GB             0B        
person-tracker-base:latest              5dc74f6a6895       1.55GB             0B        
postgres:16-alpine                      de3a4eab8fdf        294MB             0B        
redis:7.2-alpine                        b6636bae9624       38.6MB             0B        
rediscommander/redis-commander:latest   778af9bd6397       77.8MB             0B    




## server

bash -c "$(curl -fsSL https://raw.githubusercontent.com/community-scripts/ProxmoxVE/main/ct/docker.sh)"


ssh root@192.168.0.53

root
12345




#### not work maybe old docker client
docker save person-tracker-base:latest meter-watch-person-detector:latest meter-watch-cnn-recognizer:latest meter-watch-frontend:latest | pv | ssh root@192.168.0.53 'docker load'


docker save person-tracker-base:latest  | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-cnn-recognizer:latest | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-person-detector:latest | pv | ssh root@192.168.0.53 'docker load'
docker save meter-watch-frontend:latest | pv | ssh root@192.168.0.53 'docker load'






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




root@docker2:~/meter-watch# docker volume ls | grep grafana
local     meter-watch_grafana_data


docker volume rm meter-watch_grafana_data
