conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda

conda activate YOLO_t0


tensorboard --logdir=runs

docker compose --profile tools up -d


docker ps


# Посмотреть логи контейнера
docker logs redis_commander

# Или последние логи
docker logs --tail 50 redis_commander



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



docker exec -it person-detector-app /bin/bash
docker exec -it cnn-recognition-app /bin/bash
docker exec -it person_tracker_redis /bin/bash
docker exec -it redis_commander /bin/bash



docker compose build 2>&1 | tee ./build.log && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


docker compose build && (echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a"; sleep 0.1; echo -e "\a"; sleep 0.2; echo -e "\a") && echo "✅ Done!" || echo "❌ Failed!"


docker compose up --build


# run only redis
docker compose up redis





apt-get update && apt-get install -y iputils-ping
ping -c 3 192.168.0.102



