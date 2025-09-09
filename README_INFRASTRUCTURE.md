# Medical Inventory Detection - Infrastructure Setup

This document provides comprehensive setup instructions for deploying the Medical Inventory Detection system in production environments.

## 🏗️ Architecture Overview

The system consists of:
- **API Server**: FastAPI-based REST API for object detection
- **WebSocket Server**: Real-time streaming and detection
- **Model Service**: YOLO model inference engine
- **Database**: PostgreSQL for logging and metrics
- **Cache**: Redis for session management and caching
- **Monitoring**: Prometheus + Grafana for observability

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/galenzo17/inventory_check.git
cd inventory_check

# Build and start services
docker-compose -f docker/docker-compose.yml up -d

# Check service health
docker-compose ps
```

### Individual Service Deployment

```bash
# Build API image
docker build -f docker/Dockerfile.api -t medical-inventory/api:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -e REDIS_URL=redis://redis:6379 \
  -v $(pwd)/models:/app/models \
  medical-inventory/api:latest
```

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

### GPU Node Setup

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -l nvidia.com/gpu=true
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f kubernetes/

# Check deployment status
kubectl get pods -l app=medical-inventory-api
kubectl logs -f deployment/medical-inventory-api
```

### Auto-scaling Configuration

The HPA is configured to scale based on:
- CPU utilization > 70%
- Memory utilization > 80%
- Requests per second > 100

```bash
# Monitor scaling
kubectl get hpa medical-inventory-api-hpa -w
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# Custom metrics exposed
- detection_requests_total
- detection_latency_seconds
- model_inference_duration
- gpu_utilization_percent
- objects_detected_per_class
```

### Grafana Dashboards

Pre-configured dashboards available:
1. **API Performance**: Request rates, latency, error rates
2. **Model Metrics**: Inference time, accuracy, throughput
3. **Infrastructure**: CPU, memory, GPU utilization
4. **Business Metrics**: Objects detected, success rates

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000
# Open http://localhost:3000 (admin/admin)
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Required
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/medical_inventory

# Optional
MODEL_CACHE_SIZE=4
MAX_BATCH_SIZE=50
LOG_LEVEL=INFO
GPU_MEMORY_FRACTION=0.8
```

### Secrets Management

```bash
# Create secrets
kubectl create secret generic medical-inventory-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=api-key="your-secret-key"

# Use with Kubernetes
kubectl apply -f kubernetes/secrets.yaml
```

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow

The pipeline includes:
1. **Testing**: Unit tests, integration tests, security scans
2. **Building**: Docker image build and push
3. **Deployment**: Automated deployment to staging/production
4. **Monitoring**: Health checks and smoke tests

```bash
# Setup required secrets in GitHub
KUBECONFIG_STAGING: base64-encoded kubeconfig
KUBECONFIG_PRODUCTION: base64-encoded kubeconfig
SLACK_WEBHOOK: Slack notification URL
```

### Manual Deployment

```bash
# Build and tag
docker build -f docker/Dockerfile.api -t medical-inventory/api:v1.0.0 .
docker tag medical-inventory/api:v1.0.0 ghcr.io/yourorg/medical-inventory:v1.0.0

# Push to registry
docker push ghcr.io/yourorg/medical-inventory:v1.0.0

# Update Kubernetes deployment
kubectl set image deployment/medical-inventory-api api=ghcr.io/yourorg/medical-inventory:v1.0.0
kubectl rollout status deployment/medical-inventory-api
```

## 🔒 Security Considerations

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: medical-inventory-network-policy
spec:
  podSelector:
    matchLabels:
      app: medical-inventory-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

### RBAC Configuration

```bash
# Create service account with minimal permissions
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: medical-inventory-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: medical-inventory-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
EOF
```

## 📈 Scaling Guidelines

### Vertical Scaling (Resource Limits)

```yaml
resources:
  requests:
    memory: "4Gi"    # Minimum for model loading
    cpu: "2"         # 2 CPU cores
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"    # Maximum memory usage
    cpu: "4"         # Maximum CPU cores
    nvidia.com/gpu: 1
```

### Horizontal Scaling (Pod Count)

- **Minimum replicas**: 3 (high availability)
- **Maximum replicas**: 20 (cost optimization)
- **Scale up**: 50% increase when thresholds exceeded
- **Scale down**: 10% decrease with 5-minute stabilization

## 🔧 Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check GPU availability
   kubectl describe node <node-name> | grep nvidia.com/gpu
   
   # Verify device plugin
   kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset
   ```

2. **Model loading errors**
   ```bash
   # Check model storage
   kubectl exec deployment/medical-inventory-api -- ls -la /app/models
   
   # Verify PVC mount
   kubectl describe pvc model-storage-pvc
   ```

3. **High latency**
   ```bash
   # Check resource usage
   kubectl top pods -l app=medical-inventory-api
   
   # Review metrics
   curl http://localhost:9090/api/v1/query?query=detection_latency_seconds
   ```

### Performance Tuning

1. **Model Optimization**
   - Use TensorRT for GPU inference
   - Enable mixed precision (FP16)
   - Batch inference for better throughput

2. **Resource Allocation**
   - Pin CPU cores for inference
   - Use dedicated GPU nodes
   - Optimize memory allocation

3. **Caching Strategy**
   - Cache model weights in memory
   - Use Redis for session data
   - Implement response caching

## 📞 Support

For infrastructure issues:
1. Check the monitoring dashboard
2. Review pod logs: `kubectl logs -f deployment/medical-inventory-api`
3. Monitor resource usage: `kubectl top nodes`
4. Create GitHub issue with deployment logs

## 🔄 Backup and Recovery

### Database Backups

```bash
# Automated backup script
kubectl create configmap backup-script --from-file=backup.sh
kubectl create cronjob postgres-backup --image=postgres:15 --schedule="0 2 * * *" -- /backup.sh
```

### Model Versioning

```bash
# Store models with versions
/app/models/
├── v1.0.0/
│   ├── medical_nano.pt
│   └── medical_medium.pt
└── v1.1.0/
    ├── medical_nano.pt
    └── medical_medium.pt
```

This infrastructure setup provides a production-ready deployment with monitoring, scaling, and security best practices.