pip list | grep ultralytics

pip show ultralytics

pip show face_recognition


pip uninstall face_recognition



https://github.com/ageitgey/face_recognition/issues/608





for env in $(conda env list | grep -v '#' | awk '{print $1}' | grep -v 'base'); do
    echo "=== Checking environment: $env ==="
    conda run -n $env pip show ultralytics 2>/dev/null || echo "❌ Not installed"
    echo ""
done



conda create -n YOLO_t0 --clone YOLO

pip install --no-deps face_recognition
pip install --no-deps dlib

pip install git+https://github.com/ageitgey/face_recognition_models


pip install --upgrade setuptools


pip install "setuptools<81"











=== Checking environment: YOLO ===
Name: ultralytics
Version: 8.4.36
Summary: Ultralytics YOLO 🚀 for SOTA object detection, multi-object tracking, instance segmentation, pose estimation and image classification.
Home-page: https://ultralytics.com
Author: 
Author-email: Glenn Jocher <glenn.jocher@ultralytics.com>, Jing Qiu <jing.qiu@ultralytics.com>
License: AGPL-3.0
Location: /home/vadim/miniconda3/envs/YOLO/lib/python3.10/site-packages
Requires: matplotlib, numpy, opencv-python, pillow, polars, psutil, pyyaml, requests, scipy, torch, torchvision, ultralytics-thop
Required-by: 

=== Checking environment: anylabeling ===
❌ Not installed

=== Checking environment: face_recog ===
❌ Not installed

=== Checking environment: langchain_env ===
❌ Not installed

=== Checking environment: mlflow-ui-env ===
❌ Not installed

=== Checking environment: onnx_convert ===
❌ Not installed

=== Checking environment: tfjs ===
❌ Not installed

=== Checking environment: word_recognition ===
❌ Not installed

=== Checking environment: /mnt/ntfs/learn_ML/test_classes/Тестовое ===
❌ Not installed





conda activate YOLO