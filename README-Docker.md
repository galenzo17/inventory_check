# Docker Deployment Guide

This guide covers Docker-based deployment of the Medical Inventory Detection API with multi-stage builds, GPU support, and production-ready configurations.

## Quick Start

### CPU-Only Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
./docker-build.sh latest production
docker run -p 8000:8000 medical-inventory:latest
```

### GPU-Enabled Deployment

```bash
# Requires NVIDIA Docker runtime
docker-compose -f docker-compose.gpu.yml up -d

# Or build GPU-optimized image
./docker-build.sh latest gpu-production
docker run --gpus all -p 8000:8000 medical-inventory:latest
```

## Build Targets

The Dockerfile uses multi-stage builds for optimization:

- **base**: Base CUDA environment with system dependencies
- **dependencies**: Python dependencies installation
- **development**: Full development environment with testing tools
- **production-build**: Production build with only necessary files
- **production**: Minimal production runtime
- **gpu-production**: GPU-optimized production runtime

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://redis:6379` |
| `DATABASE_URL` | PostgreSQL connection URL | See docker-compose.yml |
| `MODEL_PATH` | Path to model files | `/app/models` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WORKERS` | Number of worker processes | `4` (CPU), `2` (GPU) |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |

### Volume Mounts

- `./models:/app/models:ro` - Model files (read-only)
- `./logs:/app/logs` - Application logs
- `./uploads:/app/uploads` - Temporary file uploads

## Services

### Core Services

- **api-cpu/api-gpu**: Main API service
- **websocket/websocket-gpu**: WebSocket service for real-time detection
- **redis**: Caching and rate limiting
- **postgres**: Request logging and analytics

### Monitoring Services

- **prometheus**: Metrics collection (port 9090)
- **grafana**: Monitoring dashboards (port 3000)
- **nginx**: Reverse proxy and load balancer (port 80/443)

## Production Deployment

### Security Considerations

1. **Change default passwords**:
   ```bash
   export DB_PASSWORD="your-secure-password"
   export GRAFANA_PASSWORD="your-grafana-password"
   ```

2. **Enable HTTPS**:
   - Uncomment HTTPS server block in `nginx/nginx.conf`
   - Add SSL certificates to `nginx/ssl/`

3. **Network security**:
   - Use Docker secrets for sensitive data
   - Configure firewall rules
   - Limit exposed ports

### Scaling

#### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  api-cpu:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

#### Load Balancing

Nginx automatically load balances between multiple API instances. Update `nginx/nginx.conf` upstream configuration:

```nginx
upstream api_backend {
    least_conn;
    server api-cpu-1:8000;
    server api-cpu-2:8000;
    server api-cpu-3:8000;
}
```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps
curl http://localhost/health
```

## Development

### Development Environment

```bash
# Run development container
docker build --target development -t medical-inventory:dev .
docker run -it -v $(pwd):/app -p 8000:8000 medical-inventory:dev bash

# Or use docker-compose for development
docker-compose -f docker-compose.yml up api-cpu
```

### Testing in Container

```bash
# Run tests in container
docker run --rm medical-inventory:dev pytest tests/ -v

# Run with coverage
docker run --rm medical-inventory:dev pytest tests/ --cov=. --cov-report=html
```

## Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

Available metrics:
- Request counts and latencies
- Model performance metrics
- System resource usage
- Error rates

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123)

Pre-configured dashboards:
- API Performance Dashboard
- Model Usage Analytics
- System Health Overview

## Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check NVIDIA Docker runtime
   docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
   ```

2. **Memory issues**:
   ```bash
   # Adjust container memory limits
   docker-compose up --memory=4g
   ```

3. **Model loading errors**:
   ```bash
   # Check model files exist
   docker run --rm -v $(pwd)/models:/models alpine ls -la /models
   ```

4. **Permission issues**:
   ```bash
   # Fix file permissions
   sudo chown -R 1000:1000 logs/ uploads/
   ```

### Logs

```bash
# View service logs
docker-compose logs -f api-cpu
docker-compose logs -f websocket

# View all logs
docker-compose logs -f
```

## Performance Optimization

### GPU Memory Management

```yaml
# Limit GPU memory usage
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Container Resources

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
    reservations:
      memory: 2G
      cpus: '1'
```

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL database
docker-compose exec postgres pg_dump -U inventory_user medical_inventory > backup.sql

# Restore database
docker-compose exec -T postgres psql -U inventory_user medical_inventory < backup.sql
```

### Redis Backup

```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp medical_inventory_redis:/data/dump.rdb ./redis_backup.rdb
```

## Updates and Maintenance

### Rolling Updates

```bash
# Build new image
./docker-build.sh latest production

# Update services with zero downtime
docker-compose up -d --no-deps api-cpu
```

### Cleanup

```bash
# Remove unused images and containers
docker system prune -a

# Clean up volumes (careful!)
docker volume prune
```