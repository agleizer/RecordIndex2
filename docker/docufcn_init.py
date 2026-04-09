"""
Script de inicialização do doc-UFCN.
Executado pelo serviço docufcn-init no docker-compose.
Baixa o modelo 'generic-historical-line' para o volume persistente (HF_HOME).
Na segunda execução, o huggingface_hub detecta o cache e pula o download.
"""
import sys

try:
    from doc_ufcn import models
except ImportError as e:
    print(f"[ERRO] Falha ao importar doc_ufcn: {e}", flush=True)
    print("Verifique se doc-ufcn foi instalado corretamente no Dockerfile.", flush=True)
    sys.exit(1)

print("Baixando modelo doc-UFCN 'generic-historical-line'...", flush=True)
try:
    model_path, parameters = models.download_model("generic-historical-line")
    print(f"Modelo pronto: {model_path}", flush=True)
    print(f"Classes: {parameters.get('classes')}", flush=True)
except Exception as e:
    print(f"[ERRO] Falha ao baixar modelo: {e}", flush=True)
    sys.exit(1)

print("doc-UFCN pronto.", flush=True)
