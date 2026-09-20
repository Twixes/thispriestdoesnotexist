FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/model CPU_THREADS=2 PORT=8080
WORKDIR /app
COPY inference/requirements.txt /app/inference/requirements.txt
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r inference/requirements.txt
COPY research/vendor/stylegan2-ada-pytorch /app/research/vendor/stylegan2-ada-pytorch
COPY research/THIRD_PARTY.md /app/research/THIRD_PARTY.md
COPY LICENSE /app/LICENSE
COPY inference /app/inference
RUN useradd --uid 10001 --create-home inference
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=8).read()"
CMD ["python", "-m", "inference.server"]
