#!/bin/bash
# Kubernetes deployment script for Medical Inventory Detection API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-development}
NAMESPACE="medical-inventory"
CONTEXT=${2:-$(kubectl config current-context)}

echo -e "${BLUE}Deploying Medical Inventory Detection API to Kubernetes...${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Context: ${CONTEXT}${NC}"
echo -e "${BLUE}Namespace: ${NAMESPACE}${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if kustomize is available
if ! command -v kustomize &> /dev/null; then
    echo -e "${YELLOW}Warning: kustomize not found, using kubectl apply -k${NC}"
    KUSTOMIZE_CMD="kubectl apply -k"
else
    KUSTOMIZE_CMD="kustomize build"
fi

# Verify cluster connection
echo -e "${YELLOW}Verifying cluster connection...${NC}"
if ! kubectl cluster-info --context=${CONTEXT} > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    echo -e "${RED}Context: ${CONTEXT}${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Connected to cluster${NC}"

# Create namespace if it doesn't exist
echo -e "${YELLOW}Creating namespace if needed...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Set context namespace
kubectl config set-context --current --namespace=${NAMESPACE}

# Apply GPU device plugin if needed (for GPU deployments)
if [[ "${ENVIRONMENT}" == "production" ]] || [[ "${ENVIRONMENT}" == "gpu" ]]; then
    echo -e "${YELLOW}Checking for NVIDIA device plugin...${NC}"
    if ! kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset > /dev/null 2>&1; then
        echo -e "${YELLOW}Installing NVIDIA device plugin...${NC}"
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    else
        echo -e "${GREEN}✅ NVIDIA device plugin already installed${NC}"
    fi
fi

# Deploy based on environment
case "${ENVIRONMENT}" in
    "development")
        echo -e "${YELLOW}Deploying development environment...${NC}"
        if [[ "${KUSTOMIZE_CMD}" == "kustomize build" ]]; then
            kustomize build overlays/development | kubectl apply -f -
        else
            kubectl apply -k overlays/development
        fi
        ;;
    "staging")
        echo -e "${YELLOW}Deploying staging environment...${NC}"
        if [[ "${KUSTOMIZE_CMD}" == "kustomize build" ]]; then
            kustomize build overlays/staging | kubectl apply -f -
        else
            kubectl apply -k overlays/staging
        fi
        ;;
    "production")
        echo -e "${YELLOW}Deploying production environment...${NC}"
        echo -e "${RED}WARNING: This will deploy to production!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [[ "${KUSTOMIZE_CMD}" == "kustomize build" ]]; then
                kustomize build overlays/production | kubectl apply -f -
            else
                kubectl apply -k overlays/production
            fi
        else
            echo -e "${YELLOW}Deployment cancelled${NC}"
            exit 0
        fi
        ;;
    "base")
        echo -e "${YELLOW}Deploying base configuration...${NC}"
        if [[ "${KUSTOMIZE_CMD}" == "kustomize build" ]]; then
            kustomize build base | kubectl apply -f -
        else
            kubectl apply -k base
        fi
        ;;
    *)
        echo -e "${RED}Error: Unknown environment '${ENVIRONMENT}'${NC}"
        echo -e "${YELLOW}Available environments: development, staging, production, base${NC}"
        exit 1
        ;;
esac

# Wait for deployments to be ready
echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment --all -n ${NAMESPACE}

# Check pod status
echo -e "${YELLOW}Checking pod status...${NC}"
kubectl get pods -n ${NAMESPACE} -o wide

# Show services
echo -e "${YELLOW}Services:${NC}"
kubectl get services -n ${NAMESPACE}

# Show ingress if available
if kubectl get ingress -n ${NAMESPACE} > /dev/null 2>&1; then
    echo -e "${YELLOW}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
fi

# Display access information
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${BLUE}Access Information:${NC}"

# Get LoadBalancer IP or NodePort
API_SERVICE=$(kubectl get service nginx-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
if [[ -z "${API_SERVICE}" ]]; then
    API_SERVICE=$(kubectl get service nginx-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
fi

if [[ -z "${API_SERVICE}" ]]; then
    # Try NodePort
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || echo "")
    if [[ -z "${NODE_IP}" ]]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "localhost")
    fi
    NODE_PORT=$(kubectl get service nginx-service -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
    if [[ -n "${NODE_PORT}" ]]; then
        API_SERVICE="${NODE_IP}:${NODE_PORT}"
    fi
fi

if [[ -n "${API_SERVICE}" ]]; then
    echo -e "${BLUE}  API: http://${API_SERVICE}/api/v1${NC}"
    echo -e "${BLUE}  Health: http://${API_SERVICE}/health${NC}"
    echo -e "${BLUE}  Docs: http://${API_SERVICE}/docs${NC}"
else
    echo -e "${YELLOW}  Use port-forward to access services:${NC}"
    echo -e "${BLUE}  kubectl port-forward service/nginx-service 8080:80 -n ${NAMESPACE}${NC}"
    echo -e "${BLUE}  Then access: http://localhost:8080${NC}"
fi

# Show monitoring access
PROMETHEUS_PORT=$(kubectl get service prometheus-service -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")
GRAFANA_PORT=$(kubectl get service grafana-service -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")

if [[ -n "${PROMETHEUS_PORT}" ]]; then
    echo -e "${BLUE}  Prometheus: kubectl port-forward service/prometheus-service ${PROMETHEUS_PORT}:${PROMETHEUS_PORT} -n ${NAMESPACE}${NC}"
fi

if [[ -n "${GRAFANA_PORT}" ]]; then
    echo -e "${BLUE}  Grafana: kubectl port-forward service/grafana-service ${GRAFANA_PORT}:${GRAFANA_PORT} -n ${NAMESPACE}${NC}"
fi

echo -e "${GREEN}Deployment complete!${NC}"