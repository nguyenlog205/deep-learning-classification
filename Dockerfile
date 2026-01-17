FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir fastapi uvicorn python-multipart numpy==2.2.6 \
    opencv-python pandas Pillow PyYAML scikit-learn tqdm ultralytics
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN python3 -c "from torchvision import models; models.efficientnet_b0(weights='DEFAULT')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]