FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema necessárias para Pillow e processamento de imagem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 1. PyTorch CPU primeiro — evita que o pip tente baixar a versão CUDA (~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 2. Demais dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt
# 3. doc-ufcn sem suas deps declaradas — pina numpy==1.21.2 (conflito); em runtime
#    usa o numpy moderno já instalado acima sem problema
RUN pip install --no-cache-dir --no-deps doc-ufcn

# src/ é montado como volume em dev; copiado em produção
COPY src/ ./src/

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
