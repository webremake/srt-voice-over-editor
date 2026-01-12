import json
import pysrt
import re
import os
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# ---------- Настройка Gemini API ----------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    # Если ключа нет в окружении, это будет ошибкой при запуске
    pass

genai.configure(api_key=API_KEY)
# Используем gemma-3-27b-it как одну из самых мощных открытых моделей через API
MODEL_NAME = "models/gemma-3-27b-it" 
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_PROMPT = """
You are a subtitle voice-over editor for educational videos.

TASK:
You will receive a numbered list of subtitle lines.
Your job is to edit each line to make it concise, clear, and suitable for spoken narration.

FORMAT RULES:
1. Return a numbered list matching the input exactly (e.g., if you get 1-20, return 1-20).
2. DO NOT merge lines. Each numbered line must stay separate.
3. DO NOT translate (keep original language).
4. DO NOT change terminology.
5. NO CHATTER. Start immediately with "1. [text]".
"""

# ---------- Основной скрипт ----------

def process_batch(batch_texts):
    """Отправляет батч в Gemini API и парсит нумерованный ответ."""
    prompt_lines = [f"{idx+1}. {text}" for idx, text in enumerate(batch_texts)]
    prompt_text = "\n".join(prompt_lines)
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nLines to edit:\n{prompt_text}"

    try:
        response = model.generate_content(full_prompt)
        if not response or not response.text:
            return []
            
        content = response.text.strip()
        
        # Парсинг нумерованного списка
        results = []
        for line in content.split("\n"):
            line = line.strip()
            # Ищем формат "1. Текст" или "1) Текст"
            match = re.match(r"^\d+[\.\)]\s*(.*)", line)
            if match:
                results.append(match.group(1).strip())
        
        return results
    except Exception as e:
        print(f"  [ERROR] Ошибка Gemini API ({MODEL_NAME}): {e}")
        return []

def process_srt(input_srt: Path, output_srt: Path):
    if not API_KEY:
        print("ERROR: Переменная окружения GEMINI_API_KEY не установлена.")
        return

    # 1. Загрузка файла через pysrt
    try:
        subs = pysrt.open(str(input_srt), encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Не удалось открыть файл {input_srt}: {e}")
        return

    total_blocks = len(subs)
    print(f"Всего блоков для обработки: {total_blocks}")

    chunk_size = 20
    
    # 2. Обработка батчами
    i = 0
    while i < total_blocks:
        chunk = subs[i:i+chunk_size]
        batch_ids = f"{i+1}-{min(i+chunk_size, total_blocks)}"
        print(f"Обработка батча {batch_ids} (размер {len(chunk)})...")
        
        batch_texts = [sub.text for sub in chunk]
        batch_res = process_batch(batch_texts)
        
        # 3. Логика RETRY
        if len(batch_res) != len(chunk):
            print(f"  [WARN] Несоответствие: ожидалось {len(chunk)}, получено {len(batch_res)}. Уменьшаем батч...")
            
            # Если большой батч (20) не сработал, разбиваем его на мелкие (по 5)
            batch_res = []
            sub_chunk_size = 5
            for j in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[j:j+sub_chunk_size]
                sub_texts = [sub.text for sub in sub_chunk]
                sub_ids = f"{i+j+1}-{min(i+j+sub_chunk_size, total_blocks)}"
                print(f"    Пробуем под-батч {sub_ids}...")
                
                # До 2 попыток на мелкий батч
                sub_res = []
                for attempt in range(2):
                    sub_res = process_batch(sub_texts)
                    if len(sub_res) == len(sub_chunk):
                        break
                    print(f"      [RETRY] Попытка {attempt+1} не удалась (получено {len(sub_res)}).")
                
                # Если все равно не вышло, дополняем оригиналами
                while len(sub_res) < len(sub_chunk):
                    idx_to_add = len(sub_res)
                    sub_res.append(sub_texts[idx_to_add])
                
                batch_res.extend(sub_res[:len(sub_chunk)])
            
        # Записываем результаты
        for idx, edited_text in enumerate(batch_res):
            if i + idx < total_blocks:
                subs[i + idx].text = edited_text
        
        i += chunk_size

    # 4. Сохранение
    try:
        subs.save(str(output_srt), encoding='utf-8')
        print(f"Готово! Результат сохранен в {output_srt}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения: {e}")

if __name__ == "__main__":
    input_file = Path("input.srt")
    output_file = Path("output_voiceover.srt")
    process_srt(input_file, output_file)