#!/bin/bash
# 确保输出目录存在
mkdir -p data_files/graphml

# 定义.bench文件所在目录（Windows路径转换为WSL可识别的路径）
BENCH_DIR="/mnt/d/Users/cy/Desktop/postStudy/OpenABC-master/bench_openabcd"

# 定义andAIG2Graphml.py脚本的路径
CONVERTER_SCRIPT="/mnt/d/Users/cy/Desktop/postStudy/OpenABC-master/datagen/utilities/andAIG2Graphml.py"

# 遍历所有.bench文件并转换
for bench_file in "$BENCH_DIR"/*.bench; do
    if [ -f "$bench_file" ]; then  # 确保是文件而非目录
        echo "Processing $bench_file..."
        python "$CONVERTER_SCRIPT" --bench "$bench_file" --gml data_files/graphml
    fi
done