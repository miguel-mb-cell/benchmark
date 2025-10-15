import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carrega a chave de API do seu arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configura o cliente da API
genai.configure(api_key=GOOGLE_API_KEY)

print("Buscando modelos de visão disponíveis para sua chave de API...\n")

# Itera sobre todos os modelos disponíveis
for model in genai.list_models():
  # Verifica se o modelo suporta o método 'generateContent', que é usado para visão
  if 'generateContent' in model.supported_generation_methods:
    print(f"--------------------------------------------------")
    print(f"Nome de exibição: {model.display_name}")
    print(f"Descrição: {model.description}")
    print(f"Nome para usar no código: {model.name}\n")