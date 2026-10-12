FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /gen_attack

# CPU-only PyTorch keeps the image small; use Dockerfile.gpu for CUDA.
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY . .

ENTRYPOINT ["gen-attack"]
CMD ["--help"]
