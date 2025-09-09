# Kubernetes Deployment Guide

This guide covers Kubernetes deployment of the Medical Inventory Detection API with auto-scaling, GPU support, monitoring, and production-ready configurations.

## Quick Start

### Prerequisites

- Kubernetes cluster (1.21+)
- kubectl configured
- kustomize (optional, kubectl has built-in support)
- NVIDIA GPU operator (for GPU deployments)

### Deploy to Development

```bash
cd k8s
./deploy.sh development
```

### Deploy to Production

```bash
cd k8s
./deploy.sh production
```

## Architecture Overview

The Kubernetes deployment includes:

- **API Service**: Main REST API with horizontal auto-scaling
- **API GPU Service**: GPU-optimized API instances  
- **WebSocket Service**: Real-time detection streaming
- **Database**: PostgreSQL with persistent storage
- **Cache**: Redis for session management and rate limiting
- **Proxy**: Nginx for load balancing and SSL termination
- **Monitoring**: Prometheus + Grafana stack

## Directory Structure

```
k8s/
├── base/                      # Base configurations
│   ├── namespace.yaml         # Namespace and resource quotas
│   ├── configmap.yaml         # Configuration data
│   ├── secrets.yaml           # Sensitive data templates
│   ├── persistent-volumes.yaml # Storage configurations
│   ├── redis.yaml             # Redis cache
│   ├── postgres.yaml          # PostgreSQL database
│   ├── api.yaml               # CPU-based API service
│   ├── api-gpu.yaml           # GPU-based API service
│   ├── websocket.yaml         # WebSocket service
│   ├── nginx.yaml             # Reverse proxy and ingress
│   ├── monitoring.yaml        # Prometheus and Grafana
│   └── kustomization.yaml     # Base kustomization
├── overlays/
│   ├── development/           # Development environment
│   ├── staging/               # Staging environment
│   └── production/            # Production environment
└── deploy.sh                  # Deployment script
```

## Environment Configurations

### Development
- Minimal resource allocation
- Single replicas for most services
- Debug logging enabled
- Local storage classes

### Staging  
- Moderate resource allocation
- Multiple replicas for testing
- Production-like configuration
- Staging-specific secrets

### Production
- High availability configuration
- Auto-scaling enabled
- GPU support included
- Security policies enforced
- Resource quotas and limits
- Network policies

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

The deployment includes HPA configurations for automatic scaling based on:

#### API Service
- **CPU Target**: 70%
- **Memory Target**: 80%  
- **Min Replicas**: 2
- **Max Replicas**: 20
- **Scale Up**: Fast (100% in 30s, max 4 pods in 60s)
- **Scale Down**: Conservative (50% in 60s after 5min stabilization)

#### GPU API Service
- **CPU Target**: 60%
- **Memory Target**: 75%
- **Min Replicas**: 1
- **Max Replicas**: 4
- **Scale Up**: Moderate (100% in 60s, max 1 pod in 120s)
- **Scale Down**: Very conservative (25% in 120s after 10min stabilization)

#### WebSocket Service
- **CPU Target**: 60%
- **Memory Target**: 70%
- **Min Replicas**: 1
- **Max Replicas**: 10
- **Session Affinity**: Enabled for WebSocket connections

### Vertical Pod Autoscaler (VPA)

To enable VPA for automatic resource recommendations:

```bash
# Install VPA (if not already installed)
kubectl apply -f https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler/deploy

# Enable VPA for API deployment
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-vpa
  namespace: medical-inventory
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  updatePolicy:
    updateMode: "Auto"
EOF
```

## GPU Support

### Prerequisites

1. **NVIDIA GPU Operator**:
   ```bash
   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm repo update
   helm install gpu-operator nvidia/gpu-operator -n gpu-operator-resources --create-namespace
   ```

2. **Node Labels**:
   ```bash
   kubectl label nodes <gpu-node> accelerator=nvidia-tesla-gpu
   ```

### GPU Deployment Features

- **Resource Requests**: 1 GPU per pod
- **Node Selection**: Automatic GPU node scheduling
- **Tolerations**: GPU taint tolerance
- **Memory**: Higher memory allocation (8GB limit)
- **Scaling**: Conservative scaling for expensive GPU resources

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at: `kubectl port-forward service/prometheus-service 9090:9090`

**Available Metrics**:
- API request rates and latencies
- Model inference performance
- Resource utilization
- Error rates and success rates
- Custom business metrics

### Grafana Dashboards  

Access Grafana at: `kubectl port-forward service/grafana-service 3000:3000`

**Default Credentials**: admin/admin123

**Pre-configured Dashboards**:
- API Performance Overview
- Model Usage Analytics
- Infrastructure Monitoring
- Business Metrics

### Service Discovery

Prometheus uses Kubernetes service discovery to automatically detect:
- API service endpoints
- WebSocket service endpoints
- Redis and PostgreSQL instances
- Custom application metrics

## Storage Configuration

### Persistent Volumes

The deployment uses multiple storage classes:

- **fast-ssd**: High-performance storage for databases and models
- **standard**: Standard storage for logs and temporary data

**Storage Allocations**:
- PostgreSQL: 10GB (production: 50GB)
- Redis: 1GB  
- Models: 20GB (read-only shared volume)
- Prometheus: 5GB (production: 20GB)
- Grafana: 2GB
- Logs: 5GB (shared across services)

### Model Storage

Models are stored in a shared `ReadOnlyMany` volume:
- Mounted at `/app/models` in all API pods
- Supports multiple model versions
- Hot-swappable without pod restarts
- Backup and versioning support

## Security

### Pod Security Policies

Production deployment includes:
- **Non-root execution**: All containers run as non-root users
- **Read-only root filesystem**: Where possible
- **Security contexts**: Restricted capabilities
- **AppArmor/SELinux**: Security profiles

### Network Policies

- **Ingress Control**: Only allowed traffic from nginx and monitoring
- **Egress Control**: Restricted external access
- **Service-to-Service**: Internal communication only
- **DNS Resolution**: Allowed for service discovery

### Secrets Management

- **Kubernetes Secrets**: For sensitive configuration
- **External Secrets Operator**: For cloud secret management
- **Secret Rotation**: Automated secret updates
- **Encryption at Rest**: etcd encryption enabled

## High Availability

### Database High Availability

For production PostgreSQL HA:

```bash
# Deploy PostgreSQL HA with Patroni
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres-ha bitnami/postgresql-ha -n medical-inventory
```

### Redis High Availability

For production Redis HA:

```bash
# Deploy Redis Sentinel
helm install redis-ha bitnami/redis -n medical-inventory \
  --set sentinel.enabled=true \
  --set replica.replicaCount=2
```

### Multi-Zone Deployment

```bash
# Add zone affinity rules
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app.kubernetes.io/name: api
              topologyKey: topology.kubernetes.io/zone
EOF
```

## Operations

### Deployment Commands

```bash
# Deploy specific environment
./deploy.sh development
./deploy.sh staging  
./deploy.sh production

# Deploy with specific context
./deploy.sh production my-prod-cluster

# Deploy base configuration only
./deploy.sh base
```

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment api --replicas=10 -n medical-inventory

# Check HPA status
kubectl get hpa -n medical-inventory

# View scaling events
kubectl describe hpa api-hpa -n medical-inventory
```

### Rolling Updates

```bash
# Update API image
kubectl set image deployment/api api=medical-inventory:v2.0.0 -n medical-inventory

# Check rollout status
kubectl rollout status deployment/api -n medical-inventory

# Rollback if needed
kubectl rollout undo deployment/api -n medical-inventory
```

### Health Checks

```bash
# Check all pods
kubectl get pods -n medical-inventory

# Check service endpoints
kubectl get endpoints -n medical-inventory

# Check ingress status
kubectl get ingress -n medical-inventory

# Test API health
kubectl port-forward service/nginx-service 8080:80 -n medical-inventory
curl http://localhost:8080/health
```

### Log Aggregation

```bash
# View logs from all API pods
kubectl logs -l app.kubernetes.io/name=api -n medical-inventory --tail=100 -f

# View logs from specific pod
kubectl logs <pod-name> -n medical-inventory

# Access logs via centralized logging (if configured)
# ELK Stack, Fluentd, or cloud logging solutions
```

## Troubleshooting

### Common Issues

1. **Pod Scheduling Issues**:
   ```bash
   # Check node resources
   kubectl describe nodes
   
   # Check pod events
   kubectl describe pod <pod-name> -n medical-inventory
   ```

2. **GPU Not Available**:
   ```bash
   # Check GPU nodes
   kubectl get nodes -o yaml | grep nvidia.com/gpu
   
   # Check device plugin
   kubectl get daemonset -n kube-system | grep nvidia
   ```

3. **Storage Issues**:
   ```bash
   # Check PVC status
   kubectl get pvc -n medical-inventory
   
   # Check storage classes
   kubectl get storageclass
   ```

4. **Network Connectivity**:
   ```bash
   # Test service connectivity
   kubectl run debug --image=busybox -it --rm -- sh
   # Inside pod: nslookup api-service.medical-inventory.svc.cluster.local
   ```

### Performance Tuning

1. **Resource Optimization**:
   ```bash
   # Check resource usage
   kubectl top pods -n medical-inventory
   kubectl top nodes
   ```

2. **HPA Tuning**:
   ```bash
   # Adjust scaling parameters
   kubectl patch hpa api-hpa -n medical-inventory -p '{"spec":{"targetCPUUtilizationPercentage":60}}'
   ```

3. **Database Tuning**:
   ```bash
   # Check database performance
   kubectl exec -it postgres-0 -n medical-inventory -- psql -U inventory_user -d medical_inventory -c "SELECT * FROM pg_stat_activity;"
   ```

### Backup and Recovery

```bash
# Backup database
kubectl exec postgres-0 -n medical-inventory -- pg_dump -U inventory_user medical_inventory > backup.sql

# Backup persistent volumes
kubectl get pv
# Use volume snapshots or cloud backup solutions

# Disaster recovery
# Restore from backups and redeploy
./deploy.sh production
```

## Migration

### From Docker Compose

1. Export environment variables from docker-compose
2. Create equivalent Kubernetes secrets
3. Migrate volume mounts to PVCs
4. Deploy services incrementally

### Database Migration

```bash
# Run migrations
kubectl create job migrate --from=cronjob/db-migrate -n medical-inventory
kubectl wait --for=condition=complete job/migrate -n medical-inventory
```

This completes the comprehensive Kubernetes deployment guide with auto-scaling, GPU support, monitoring, and production-ready configurations.