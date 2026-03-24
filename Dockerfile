FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.7.1

WORKDIR /app

# System deps for scikit-image/scikit-learn
RUN pip install --no-cache-dir \
    scikit-learn \
    scikit-image \
    safetensors \
    pyyaml \
    fastapi \
    uvicorn \
    websockets

# Copy project
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3"]
