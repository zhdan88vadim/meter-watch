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

docker compose up person-detector-app


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

