#!/bin/bash

SERVER="root@192.168.0.53"
DEPLOY_PATH="/root/meter-watch"

echo "🚀 Complete deployment to $SERVER"
echo "=================================="

# Create directory on server
echo "📁 Creating directory on server..."
ssh $SERVER "mkdir -p $DEPLOY_PATH"

# 1. Build images locally
echo "📦 Building images..."
docker compose build

# 2. Create configuration package
echo "📦 Creating configuration package..."
tar -czf config.tar.gz \
    docker-compose.yml \
    .env \
    postgres/ \
    pgadmin/ \
    grafana/ \
    web/ 2>/dev/null || true

# 3. Transfer configuration
echo "📤 Transferring configuration..."
scp config.tar.gz $SERVER:$DEPLOY_PATH/

# 4. Transfer images via streaming
echo ""
echo "📤 Streaming Docker images..."
echo "=============================="

# Function to stream image
stream_image() {
    local image=$1
    if docker image inspect "$image" &>/dev/null; then
        echo "📦 Streaming $image..."
        docker save "$image" | ssh $SERVER "docker load" 2>/dev/null
        echo "  ✅ $image done"
    else
        echo "  ⚠️  $image not found locally, skipping..."
    fi
}

# Stream all needed images
# stream_image "person-tracker-base:latest"
stream_image "meter-watch-cnn-recognizer:latest"
# stream_image "meter-watch-person-detector:latest"
# stream_image "meter-watch-frontend:latest"
# stream_image "redis:7.2-alpine"
# stream_image "postgres:16-alpine"
# stream_image "dpage/pgadmin4:latest"
# stream_image "grafana/grafana:latest"
# stream_image "rediscommander/redis-commander:latest"

# 5. Deploy on server
echo ""
echo "🚀 Deploying services..."
ssh -t $SERVER << 'ENDSSH'
cd /root/meter-watch

# Extract configuration
echo "📦 Extracting configuration..."
tar -xzf config.tar.gz

# Create directories
echo "📁 Creating directories..."
mkdir -p recordings output logs postgres/init pgadmin grafana/provisioning
chmod -R 755 recordings output logs

# Check .env
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Stop old containers
echo "🛑 Stopping old containers..."
docker compose down 2>/dev/null || true

# Start services
echo "🚀 Starting services..."
docker compose up -d

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 5

# Show status
echo ""
echo "📊 Service status:"
docker compose ps

echo ""
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""
echo "🌐 Services running on 192.168.0.53:"
echo "  - Person Detector:   http://192.168.0.53:5000"
echo "  - CNN Recognizer:    http://192.168.0.53:5002"
echo "  - Frontend:          http://192.168.0.53:8080"
echo "  - Redis Commander:   http://192.168.0.53:8081"
echo "  - pgAdmin:           http://192.168.0.53:5050"
echo "  - Grafana:           http://192.168.0.53:3000"
echo ""
echo "📝 Useful commands:"
echo "  View logs:      docker compose logs -f"
echo "  Stop services:  docker compose down"
echo "  Check status:   docker compose ps"
ENDSSH

# Clean up local files
echo "🧹 Cleaning up..."
rm -f config.tar.gz

echo ""
echo "✅ Complete! Services running on 192.168.0.53"