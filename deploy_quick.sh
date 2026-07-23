#!/bin/bash

SERVER="root@192.168.0.53"
DEPLOY_PATH="/root/meter-watch"

echo "🔄 Quick update to $SERVER"
echo "=========================="

# Pull latest changes
echo "📥 Pulling latest code..."
git pull

# Build only what changed
echo "📦 Building..."
docker compose build person-detector cnn-recognizer

# Transfer images
echo "📤 Transferring images..."
docker save meter-watch-person-detector:latest | ssh $SERVER "docker load"
docker save meter-watch-cnn-recognizer:latest | ssh $SERVER "docker load"

# Transfer config files
echo "📤 Transferring config..."
scp docker-compose.yml .env $SERVER:$DEPLOY_PATH/

# Restart services
ssh -t $SERVER "cd $DEPLOY_PATH && docker compose down && docker compose up -d"

echo "✅ Update complete!"