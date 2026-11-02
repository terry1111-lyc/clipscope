FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface CLIPSCOPE_DEVICE=cpu
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir '.[model]' && \
    mkdir -p /app/data /app/.cache /app/experiments/results && chown -R appuser:appuser /app
COPY --chown=appuser:appuser experiments ./experiments
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
CMD ["clipscope", "serve", "--host", "0.0.0.0", "--port", "8000"]
