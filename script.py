import json
import pysrt
import re
import os
import google.generativeai as genai
import time
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
# Используем gemini-3-flash-preview для баланса скорости и качества
MODEL_NAME = "models/gemini-3-flash-preview" 
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_PROMPT = """
Ты — профессиональный редактор субтитров для образовательных видео.

ТВОЯ ЗАДАЧА:
Ты получишь нумерованный список строк субтитров на РУССКОМ языке.
Тебе нужно отредактировать каждую строку, чтобы она стала лаконичной, четкой и подходила для озвучки (voice-over).

ПРАВИЛА:
1. ВЫХОД СТРОГО НА РУССКОМ ЯЗЫКЕ. Ни в коем случае не переводи на английский.
2. Сохраняй нумерацию строк (1, 2, 3...). Количество строк на выходе должно в точности совпадать с входом.
3. НЕ объединяй строки. Каждая строка под своим номером — это отдельный блок.
4. Убирай лишние слова, вводные конструкции и разговорный мусор ("так сказать", "вот", "ну").
5. Пиши СТРОГО результат, без лишних вступлений и комментариев.
"""

# ---------- Основной скрипт ----------

def process_batch(batch_texts):
    """
    Отправляет батч строк в Gemini API и парсит нумерованный ответ.

    Args:
        batch_texts (list[str]): Список текстовых строк субтитров для редактирования.

    Returns:
        list[str]: Список отредактированных строк. Пустой список в случае ошибки API или парсинга.
    
    Note:
        Функция ожидает, что модель вернет нумерованный список в формате "1. Текст".
        Используется регулярное выражение для извлечения текста после номера.
    """
    prompt_lines = [f"{idx+1}. {text}" for idx, text in enumerate(batch_texts)]
    prompt_text = "\n".join(prompt_lines)
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nLines to edit:\n{prompt_text}"

    try:
        response = model.generate_content(full_prompt)
        if not response or not response.text:
            return []
            
        content = response.text.strip()
        
        # Парсинг нумерованного списка из текстового ответа модели
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
    """
    Основной цикл обработки SRT файла.
    
    Загружает файл, разбивает на батчи по 50 строк, применяет редактирование через Gemini API
    с использованием двухуровневой логики ретраев (10 строк, затем 5 строк) в случае ошибок.

    Args:
        input_srt (Path): Путь к входному SRT файлу.
        output_srt (Path): Путь для сохранения обработанного файла.

    Flow:
        1. Загрузка через pysrt (сохраняет оригинальные таймкоды).
        2. Обработка батчами по 50 блоков.
        3. Если 50 блоков не прошли (несовпадение количества строк) -> ретрай по 10 блоков.
        4. Если 10 блоков не прошли -> ретрай по 5 блоков.
        5. Если 5 блоков не прошли -> использование оригинального текста как fallback.
        6. Пауза 15 секунд между всеми API-запросами для соблюдения лимитов (5 RPM).
    """
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

    chunk_size = 50
    
    # 2. Обработка батчами
    i = 0
    while i < total_blocks:
        chunk = subs[i:i+chunk_size]
        batch_ids = f"{i+1}-{min(i+chunk_size, total_blocks)}"
        print(f"Обработка батча {batch_ids} (размер {len(chunk)})...")
        
        batch_texts = [sub.text for sub in chunk]
        batch_res = process_batch(batch_texts)
        
        # 3. Логика RETRY (10 -> 5)
        # Если количество строк в ответе не совпадает с запросом, начинаем дробление
        if len(batch_res) != len(chunk):
            print(f"  [WARN] Несоответствие: ожидалось {len(chunk)}, получено {len(batch_res)}. Пробуем батчи по 10...")
            
            batch_res = []
            sub_chunk_size = 10
            for j in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[j:j+sub_chunk_size]
                sub_texts = [sub.text for sub in sub_chunk]
                sub_ids = f"{i+j+1}-{min(i+j+sub_chunk_size, total_blocks)}"
                print(f"    Пробуем батч по 10: {sub_ids}...")
                
                # Пауза перед каждым запросом в ретрае для соблюдения Rate Limit
                time.sleep(15)
                sub_res = process_batch(sub_texts)
                
                # Если батч по 10 не прошел, разбиваем его на батчи по 5
                if len(sub_res) != len(sub_chunk):
                    print(f"      [WARN] Батч по 10 не прошел. Пробуем батчи по 5 для этого участка...")
                    sub_res = []
                    sub_sub_chunk_size = 5
                    for k in range(0, len(sub_chunk), sub_sub_chunk_size):
                        ssc = sub_chunk[k:k+sub_sub_chunk_size]
                        ssc_texts = [sub.text for sub in ssc]
                        ssc_ids = f"{i+j+k+1}-{min(i+j+k+sub_sub_chunk_size, total_blocks)}"
                        print(f"        Пробуем батч по 5: {ssc_ids}...")
                        
                        # Пауза перед каждым запросом по 5 строк
                        time.sleep(15)
                        
                        # До 2 попыток на батч по 5
                        final_res = []
                        for attempt in range(2):
                            final_res = process_batch(ssc_texts)
                            if len(final_res) == len(ssc):
                                break
                            print(f"          [RETRY] Попытка {attempt+1} (5 строк) не удалась.")
                            time.sleep(15)
                        
                        # Если 5 строк не прошли даже после ретраев, берем оригинал (fallback)
                        if len(final_res) < len(ssc):
                            while len(final_res) < len(ssc):
                                final_res.append(ssc_texts[len(final_res)])
                        
                        sub_res.extend(final_res[:len(ssc)])
                
                batch_res.extend(sub_res[:len(sub_chunk)])
        
        # Записываем результаты обратно в объекты pysrt
        for idx, edited_text in enumerate(batch_res):
            if i + idx < total_blocks:
                subs[i + idx].text = edited_text
        
        i += chunk_size
        
        # Добавляем задержку для соблюдения лимитов API (5 RPM для Gemini Free Tier)
        if i < total_blocks:
            print("Ожидание 15 секунд перед следующим основным батчем...")
            time.sleep(15)

    # 4. Сохранение результата
    try:
        subs.save(str(output_srt), encoding='utf-8')
        print(f"Готово! Результат сохранен в {output_srt}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения: {e}")

if __name__ == "__main__":
    input_file = Path("input.srt")
    output_file = Path("output_voiceover.srt")
    process_srt(input_file, output_file)