# Sử dụng base image có sẵn CUDA 12.1 và Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Thiết lập môi trường không tương tác (tránh bị dừng lại hỏi zone)
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file môi trường vào trước để tận dụng cache của Docker
COPY environment.yml .

# Cài đặt các thư viện từ environment.yml (Dùng pip cho nhanh trong Docker)
# Mình sẽ bóc tách phần pip trong yml để cài
RUN pip3 install --upgrade pip
RUN pip3 install fastapi uvicorn python-multipart numpy==2.2.6 \
    opencv-python pandas Pillow PyYAML scikit-learn tqdm ultralytics
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy toàn bộ code vào container
COPY . .

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy server khi container khởi động
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]