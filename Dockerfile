FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src/ ./src/

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

CMD ["python", "-c", "import pipeoptz; print(pipeoptz.__version__)"]