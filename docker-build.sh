#!/bin/bash
# Docker build script for Medical Inventory Detection API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="medical-inventory"
TAG=${1:-latest}
BUILD_TARGET=${2:-production}

echo -e "${BLUE}Building Medical Inventory Detection API Docker image...${NC}"
echo -e "${BLUE}Image: ${IMAGE_NAME}:${TAG}${NC}"
echo -e "${BLUE}Target: ${BUILD_TARGET}${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --target ${BUILD_TARGET} \
    --tag ${IMAGE_NAME}:${TAG} \
    --tag ${IMAGE_NAME}:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
    echo -e "${GREEN}Image: ${IMAGE_NAME}:${TAG}${NC}"
    
    # Show image size
    docker images ${IMAGE_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    echo -e "${BLUE}To run the image:${NC}"
    echo -e "${BLUE}  docker run -p 8000:8000 ${IMAGE_NAME}:${TAG}${NC}"
    echo -e "${BLUE}To run with GPU support:${NC}"
    echo -e "${BLUE}  docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${TAG}${NC}"
    echo -e "${BLUE}To run with docker-compose:${NC}"
    echo -e "${BLUE}  docker-compose up${NC}"
    
else
    echo -e "${RED}❌ Docker image build failed!${NC}"
    exit 1
fi

# Optional: Run security scan if trivy is installed
if command -v trivy &> /dev/null; then
    echo -e "${YELLOW}Running security scan...${NC}"
    trivy image ${IMAGE_NAME}:${TAG}
fi

echo -e "${GREEN}Build complete!${NC}"