import os
import json
import time
import base64
from datetime import datetime
from pathlib import Path
import subprocess
from datetime import date

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
ARQUIVO_FOTOS_ENVIADAS = CAMINHO_RESULTADOS / "fotos_enviadas.json"

def load_sent_photos():
    """Carrega as fotos já enviadas no dia atual do arquivo JSON, criando-o se não existir."""
    CAMINHO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    
    if not ARQUIVO_FOTOS_ENVIADAS.exists():
        with open(ARQUIVO_FOTOS_ENVIADAS, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
        return {}

    try:
        with open(ARQUIVO_FOTOS_ENVIADAS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Converte strings de volta para Path
            return {day: [Path(p) for p in photos] for day, photos in data.items()}
    except json.JSONDecodeError:
        print("⚠️  Arquivo 'fotos_enviadas.json' estava corrompido. Recriando.")
        with open(ARQUIVO_FOTOS_ENVIADAS, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
        return {}


def save_sent_photos(sent_photos):
    CAMINHO_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Cria arquivo vazio se não existir (segurança extra; opcional)
    if not ARQUIVO_FOTOS_ENVIADAS.exists():
        with open(ARQUIVO_FOTOS_ENVIADAS, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    # Garante que todo valor seja serializável (lista de strings)
    serializable_data = {}
    for day, photos in sent_photos.items():
        # photos pode conter Path ou str; convert para str
        serializable_data[day] = [str(p) for p in photos]

    with open(ARQUIVO_FOTOS_ENVIADAS, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)

def get_today_photos_to_send(all_photos):
    """Retorna as 5 fotos designadas para o dia atual (as mesmas em todas as execuções do dia)."""
    today = str(date.today())

    # Carrega as fotos já enviadas
    sent_photos = load_sent_photos()

    # Se já existem fotos designadas para hoje → retorna elas
    if today in sent_photos and isinstance(sent_photos[today], list) and len(sent_photos[today]) > 0:
        print(f"📷 Usando fotos já designadas para hoje ({today})")
        return [Path(p) for p in sent_photos[today]]

    # Caso contrário, selecionar novas fotos para o dia
    print(f"🆕 Selecionando novas fotos para hoje ({today})")
    already_sent = set()
    for day, photos in sent_photos.items():
        if day != today:  # Não incluir o dia de hoje na contagem de "já enviadas"
            for p in photos:
                already_sent.add(Path(p))

    # Filtra as que ainda não foram usadas em nenhum dia anterior
    available_photos = [photo for photo in all_photos if photo not in already_sent]

    if len(available_photos) == 0:
        print("⚠️  Todas as fotos já foram enviadas! Reiniciando o ciclo...")
        available_photos = all_photos

    # Seleciona até 5 novas fotos (ou menos, se não houver tantas)
    photos_for_today = available_photos[:5]

    # Salva no histórico (mantendo persistência entre execuções)
    sent_photos[today] = photos_for_today
    save_sent_photos(sent_photos)

    return photos_for_today


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
    print(f"✅ Resultado salvo para '{api_name}' em: {output_path}")

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
    print(f"  - Chamando GPT-4o...")
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
            max_completion_tokens=500
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Erro ao chamar a API: {str(e)}"
    end_time = time.time()
    return {"response": answer, "latency_seconds": end_time - start_time}


def call_gemini(image_path, pergunta):
    print(f"  - Chamando Gemini 2.5 Pro...")
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
    
    start_time = time.time()
    
    pergunta_en = call_google_translate(pergunta_pt, target_language='en')
    
    if not moondream_model:
        end_time = time.time()
        return {
            "pergunta_translated_en": pergunta_en,
            "response_original_en": "Moondream não inicializado",
            "response_translated_pt": "Moondream não inicializado",
            "latency_seconds": end_time - start_time
        }

    try:
        image = Image.open(image_path)
        result = moondream_model.query(image, pergunta_en)
        answer_en = result.get("answer", "Chave 'answer' não encontrada.")
    except Exception as e:
        answer_en = f"Erro ao chamar a API Moondream: {str(e)}"
    
    answer_pt = call_google_translate(answer_en, target_language='pt')

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
    print("=" * 60)
    print("🚀 Iniciando o script de benchmark...")
    print("=" * 60)

    # Carrega todas as fotos disponíveis
    all_photos_jpg = list(CAMINHO_IMAGENS.glob("*.jpg"))
    all_photos_jpeg = list(CAMINHO_IMAGENS.glob("*.jpeg"))
    all_photos = sorted(all_photos_jpg + all_photos_jpeg)
    print(f"📁 Total de fotos disponíveis: {len(all_photos)}")

    # Obtém as fotos do dia (sempre as mesmas para os 6 horários)
    photos_to_send_today = get_today_photos_to_send(all_photos)
    
    print(f"\n📸 Fotos selecionadas para hoje ({date.today()}):")
    for i, photo in enumerate(photos_to_send_today, 1):
        print(f"   {i}. {photo.name}")
    
    # Carrega as perguntas
    try:
        with open(ARQUIVO_PERGUNTAS, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Arquivo de perguntas não encontrado em: {ARQUIVO_PERGUNTAS}")
        return
    
    # Cria um dicionário mapeando arquivo_imagem -> perguntas
    photo_questions_map = {}
    for item in data:
        if "arquivo_imagem" in item and "pergunta" in item:
            filename = item["arquivo_imagem"]
            questions = item["pergunta"] if isinstance(item["pergunta"], list) else [item["pergunta"]]
            photo_questions_map[filename] = questions
    
    # Calcula total de combinações
    total_combinations = sum(len(photo_questions_map.get(photo.name, [])) for photo in photos_to_send_today)
    
    print(f"\n❓ Mapeamento de perguntas carregado para {len(photo_questions_map)} fotos")
    print(f"🔄 Total de combinações a processar: {total_combinations} (foto + pergunta específica)\n")
    
    # Contador de progresso
    current_combination = 0
    
    # Processa cada foto do dia
    for photo_path in photos_to_send_today:
        # Busca as perguntas específicas para esta foto
        questions_for_photo = photo_questions_map.get(photo_path.name, [])
        
        if not questions_for_photo:
            print(f"\n⚠️  Nenhuma pergunta encontrada para: {photo_path.name} - Pulando...")
            continue
        
        print(f"\n{'='*60}")
        print(f"📷 Processando foto: {photo_path.name}")
        print(f"   Total de perguntas para esta foto: {len(questions_for_photo)}")
        print(f"{'='*60}")
        
        for question in questions_for_photo:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] 💬 Pergunta: '{question}'")
            sent_timestamp = datetime.now().isoformat()

            # Chama as APIs
            gpt4o_result = call_gpt4o(photo_path, question)
            gpt5mini_result = call_gpt5_mini(photo_path, question)
            gemini_pro_result = call_gemini(photo_path, question)
            gemini_flash_result = call_gemini_flash(photo_path, question)
            moondream_result = call_moondream_with_translation(photo_path, question)

            # Estrutura e salva os dados
            base_data = {
                "arquivo_imagem": photo_path.name, 
                "question_original_pt": question, 
                "timestamp_sent_utc": sent_timestamp,
                "date": str(date.today())
            }
            
            save_result("gpt-4o", photo_path, question, {**base_data, **gpt4o_result})
            save_result("gpt-5-mini", photo_path, question, {**base_data, **gpt5mini_result})
            save_result("gemini-2.5-pro", photo_path, question, {**base_data, **gemini_pro_result})
            save_result("gemini-flash-latest", photo_path, question, {**base_data, **gemini_flash_result})
            save_result("moondream", photo_path, question, {**base_data, **moondream_result})

    print(f"\n{'='*60}")
    print(f"✅ Processamento concluído para as {len(photos_to_send_today)} fotos de hoje!")
    print(f"📊 Total de combinações processadas: {total_combinations}")
    print(f"{'='*60}\n")

    
if __name__ == "__main__":
    main()
