FROM python:3.11

# 安装依赖
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install pygame numpy pyaudio

CMD ["python", "automind.py"]
