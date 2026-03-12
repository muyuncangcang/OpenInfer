#!/bin/bash

################################################################################
# Build OpenInfer Docker Image
################################################################################

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Building OpenInfer Docker Image"
echo "========================================="
echo ""

docker build -f Dockerfile -t open_infer:latest ..

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
docker images open_infer:latest
