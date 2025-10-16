import os
import json
import time
import base64
from datetime import datetime
from pathlib import Path
import subprocess

import moondream as md

import google.generativeai as genai
from openai import OpenAI
from PIL import Image

import logging
logging.getLogger('google.api_core').setLevel(logging.ERROR)

import random

MAX_TASKS_PER_RUN = 5

from dotenv import load_dotenv

load_dotenv()
# --- CONFIGURAÇÃO ---
# Carrega as chaves de API a partir das variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

# --- INICIALIZAÇÃO DOS CLIENTES ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')
gemini_flash_model = genai.GenerativeModel('models/gemini-flash-latest')

# Inicializa o cliente do Moondream com a API Key
try:
    if MOONDREAM_API_KEY:
        moondream_model = md.vl(api_key=MOONDREAM_API_KEY)
    else:
        moondream_model = None
        print("AVISO: MOONDREAM_API_KEY não encontrada. As chamadas para o Moondream serão puladas.")
except Exception as e:
    moondream_model = None
    print(f"AVISO: Falha ao inicializar o cliente do Moondream. Erro: {e}")

# Caminhos (sem alteração)
CAMINHO_DADOS = Path("data")
CAMINHO_IMAGENS = CAMINHO_DADOS / "imagens"
ARQUIVO_PERGUNTAS = CAMINHO_DADOS / "perguntas.json"
CAMINHO_RESULTADOS = Path("resultados")
ARQUIVO_LOG = CAMINHO_RESULTADOS / "processed_log.json"


def load_processed_log():
    """Carrega o conjunto de tarefas já processadas a partir do arquivo de log."""
    if not ARQUIVO_LOG.exists():
        return set()
    with open(ARQUIVO_LOG, 'r', encoding='utf-8') as f:
        # Usamos um 'set' para buscas muito mais rápidas
        return set(json.load(f))

def save_processed_log(processed_set):
    """Salva o conjunto atualizado de tarefas processadas no arquivo de log."""
    CAMINHO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_LOG, 'w', encoding='utf-8') as f:
        # Convertemos o 'set' de volta para uma lista para salvar em JSON
        json.dump(list(processed_set), f, indent=4)


# --- FUNÇÕES DE HELPER (save_result e encode_image não são mais necessárias para o moondream) ---
def save_result(api_name, arquivo_imagem, pergunta, response_data):
    """Salva o resultado do benchmark em um arquivo JSON."""
    result_dir = CAMINHO_RESULTADOS / api_name
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pergunta_slug = "".join(filter(str.isalnum, pergunta))[:30]
    filename = f"{Path(arquivo_imagem).stem}_{pergunta_slug}_{timestamp}.json"
    output_path = result_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=4)
    print(f"Resultado salvo para '{api_name}' em: {output_path}")

def call_google_translate(text, target_language='en'):
    """Chama o script translate.js para usar a mesma biblioteca da aplicação."""
    if not text:
        return ""
    try:
        # Executa o comando: node translate.js "texto para traduzir" "pt"
        process = subprocess.run(
            ['node', 'translate.js', text, target_language],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        # Retorna a saída do script, removendo espaços em branco extras
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"  - Erro ao chamar o script de tradução Node.js: {e.stderr}")
        return f"Erro na tradução: {e.stderr}"
    except FileNotFoundError:
        print("  - Erro: 'node' não encontrado. O Node.js está instalado e no PATH?")
        return "Erro: Node.js não encontrado."


# --- FUNÇÕES DE CHAMADA DE API (GPT-4o e Gemini sem alteração) ---
def encode_image_to_base64(image_path):
    """Codifica uma imagem para o formato base64. Usado apenas por GPT-4o agora."""
    with open(image_path, "rb") as arquivo_imagem:
        return base64.b64encode(arquivo_imagem.read()).decode('utf-8')

def call_gpt4o(image_path, pergunta):
    print(f"Chamando GPT-4o para a imagem: {image_path.name}")
    base64_image = encode_image_to_base64(image_path)
    start_time = time.time()
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": pergunta}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=500)
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Erro ao chamar a API: {str(e)}"
    end_time = time.time()
    latency = end_time - start_time
    return {"response": answer, "latency_seconds": latency}

def call_gpt5_mini(image_path, pergunta):
    print(f"  - Chamando GPT-5 Mini...")
    base64_image = encode_image_to_base64(image_path)
    start_time = time.time()
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini", 
            messages=[{"role": "user", "content": [{"type": "text", "text": pergunta}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            max_completion_tokens=500 # <-- CORREÇÃO: Parâmetro atualizado de 'max_tokens'
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Erro ao chamar a API: {str(e)}"
    end_time = time.time()
    return {"response": answer, "latency_seconds": end_time - start_time}


def call_gemini(image_path, pergunta):
    print(f"Chamando Gemini 2.5 Pro para a imagem: {image_path.name}")
    start_time = time.time()
    try:
        img = Image.open(image_path)
        response = gemini_model.generate_content([pergunta, img])
        answer = response.text
    except Exception as e:
        answer = f"Erro ao chamar a API: {str(e)}"
    end_time = time.time()
    latency = end_time - start_time
    return {"response": answer, "latency_seconds": latency}


def call_gemini_flash(image_path, pergunta):
    """Chama a API Gemini Flash e cronometra a resposta."""
    print(f"  - Chamando Gemini Flash Latest...")
    start_time = time.time()
    try:
        img = Image.open(image_path)
        # Usa o objeto do modelo Flash que inicializamos
        response = gemini_flash_model.generate_content([pergunta, img])
        answer = response.text
    except Exception as e:
        answer = f"Erro ao chamar a API: {str(e)}"
    end_time = time.time()
    latency = end_time - start_time
    return {"response": answer, "latency_seconds": latency}


def call_moondream_with_translation(image_path, pergunta_pt):
    print(f"  - Chamando Moondream (com tradução)...")
    
    # <-- CORREÇÃO: Cronômetro iniciado ANTES de qualquer operação
    start_time = time.time()
    
    pergunta_en = call_google_translate(pergunta_pt, target_language='en')
    
    if not moondream_model:
        # ... (código de erro omitido por brevidade)
        return {}

    try:
        image = Image.open(image_path)
        result = moondream_model.query(image, pergunta_en)
        answer_en = result.get("answer", "Chave 'answer' não encontrada.")
    except Exception as e:
        answer_en = f"Erro ao chamar a API Moondream: {str(e)}"
    
    answer_pt = call_google_translate(answer_en, target_language='pt')

    # <-- CORREÇÃO: Cronômetro parado APÓS todas as operações
    end_time = time.time()
    latency = end_time - start_time

    return {
        "pergunta_translated_en": pergunta_en,
        "response_original_en": answer_en,
        "response_translated_pt": answer_pt,
        "latency_seconds": latency
    }


def main():
    """Orquestra o benchmark, processando um número limitado de tarefas novas por execução."""
    print("Iniciando o script de benchmark...")
    
    processed_log = load_processed_log()
    print(f"Encontradas {len(processed_log)} tarefas já processadas no log.")
    
    try:
        with open(ARQUIVO_PERGUNTAS, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Arquivo de perguntas não encontrado em: {ARQUIVO_PERGUNTAS}")
        return

    # NOVO: Monta uma lista de todas as tarefas pendentes
    tasks_to_process = []
    for item in data:
        if "arquivo_imagem" not in item or "pergunta" not in item:
            continue
        
        arquivo_imagem = item["arquivo_imagem"]
        for question in item["pergunta"]:
            unique_id = f"{arquivo_imagem}::{question}"
            if unique_id not in processed_log:
                tasks_to_process.append({
                    "unique_id": unique_id,
                    "arquivo_imagem": arquivo_imagem,
                    "question": question,
                    "image_path": CAMINHO_IMAGENS / arquivo_imagem
                })

    # NOVO: Embaralha a lista de tarefas pendentes para garantir variedade
    random.shuffle(tasks_to_process)
    
    # NOVO: Seleciona o número máximo de tarefas para esta execução
    tasks_for_this_run = tasks_to_process[:MAX_TASKS_PER_RUN]
    
    print(f"Encontradas {len(tasks_to_process)} novas tarefas. Processando no máximo {len(tasks_for_this_run)} nesta execução.")

    # ALTERADO: O loop principal agora itera sobre a pequena lista de tarefas selecionadas
    for task in tasks_for_this_run:
        # Extrai as informações da tarefa
        unique_id = task["unique_id"]
        arquivo_imagem = task["arquivo_imagem"]
        question = task["question"]
        image_path = task["image_path"]

        if not image_path.exists():
            print(f"\nAviso: Imagem '{arquivo_imagem}' não encontrada. Pulando tarefa.")
            continue

        print(f"\nProcessando nova tarefa: '{arquivo_imagem}' com a pergunta: '{question}'")
        sent_timestamp = datetime.now().isoformat()

        # Chama as APIs
        gpt4o_result = call_gpt4o(image_path, question)
        gpt5mini_result = call_gpt5_mini(image_path, question)
        gemini_pro_result = call_gemini(image_path, question)
        gemini_flash_result = call_gemini_flash(image_path, question)
        moondream_result = call_moondream_with_translation(image_path, question)

        # Estrutura e salva os dados
        base_data = {"arquivo_imagem": arquivo_imagem, "question_original_pt": question, "timestamp_sent_utc": sent_timestamp}
        save_result("gpt-4o", arquivo_imagem, question, {**base_data, **gpt4o_result})
        save_result("gpt-5-mini", arquivo_imagem, question, {**base_data, **gpt5mini_result})
        save_result("gemini-2.5-pro", arquivo_imagem, question, {**base_data, **gemini_pro_result})
        save_result("gemini-flash-latest", arquivo_imagem, question, {**base_data, **gemini_flash_result})
        save_result("moondream", arquivo_imagem, question, {**base_data, **moondream_result})

        # Adiciona o ID da tarefa ao log após o sucesso
        processed_log.add(unique_id)
        # Salva o log a cada nova tarefa processada para evitar perdas
        save_processed_log(processed_log)

    print(f"\nProcessamento concluído. {len(tasks_for_this_run)} tarefas foram processadas nesta execução.")
    
if __name__ == "__main__":
    main()
