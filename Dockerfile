FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc libgeos-dev libopenblas-base libopenmpi-dev libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade -r app_requirements.txt

RUN pip install -r hrnet_requirements.txt

RUN python tools/server.py

EXPOSE 8080

CMD ["python", "tools/server.py", "serve"]
