#!/bin/bash
set -e

# ============================================================
# 部署脚本：Mac 上构建镜像 → 导出 → 上传到 T4 服务器 → 导入
# 用法: ./deploy.sh <T4服务器IP或hostname> [用户名]
# 示例: ./deploy.sh 192.168.1.100 ubuntu
#       ./deploy.sh gpu-server root
# ============================================================

IMAGE="xbotgo-tensorrt"
TAG="latest"
TARBALL="${IMAGE}-${TAG}.tar.gz"

# --------------- 参数 ---------------
T4_HOST="${1:?用法: ./deploy.sh <T4服务器IP> [用户名]}"
T4_USER="${2:-root}"

echo "============================================"
echo " Step 1/4: 构建 Docker 镜像"
echo "============================================"
docker build -t ${IMAGE}:${TAG} .

echo ""
echo "============================================"
echo " Step 2/4: 导出镜像为 tar.gz"
echo "============================================"
echo "[INFO] 导出中，请稍候..."
docker save ${IMAGE}:${TAG} | gzip > ${TARBALL}
echo "[INFO] 导出完成: ${TARBALL} ($(du -h ${TARBALL} | cut -f1))"

echo ""
echo "============================================"
echo " Step 3/4: 上传到 T4 服务器"
echo "============================================"
scp ${TARBALL} ${T4_USER}@${T4_HOST}:~/
echo "[INFO] 上传完成"

echo ""
echo "============================================"
echo " Step 4/4: 在 T4 上导入并验证"
echo "============================================"
ssh ${T4_USER}@${T4_HOST} << 'ENDSSH'
    set -e
    IMAGE_FILE="xbotgo-tensorrt-latest.tar.gz"

    echo "[T4] 导入镜像..."
    docker load < ~/${IMAGE_FILE}
    rm -f ~/${IMAGE_FILE}

    echo ""
    echo "[T4] 验证可执行文件..."
    docker run --rm xbotgo-tensorrt:latest ls -lh /workspace/build/

    echo ""
    echo "[T4] ====== 部署完成 ======"
    echo ""
    echo "后续在 T4 上执行:"
    echo "  docker run --gpus all -it --rm \\"
    echo "      -v \$(pwd)/model:/workspace/model \\"
    echo "      xbotgo-tensorrt:latest build-engine"
ENDSSH

echo ""
echo "============================================"
echo " 本地清理临时文件"
echo "============================================"
rm -f ${TARBALL}
echo "[INFO] 已删除本地 ${TARBALL}"

echo ""
echo "===== 全部完成 ====="
