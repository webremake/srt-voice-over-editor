import json
from pathlib import Path
from pydantic import BaseModel, ValidationError, conint, constr
from datetime import timedelta
import srt
import ollama

# ---------- Pydantic модели для строгой валидации ----------
class SubtitleLine(BaseModel):
    text: constr(min_length=1)  # не пустой текст

class SubtitleBlock(BaseModel):
    index: conint(gt=0)
    start: str  # "00:00:01,000"
    end: str    # "00:00:05,000"
    lines: list[SubtitleLine]

# ---------- Функции ----------
def srt_to_json(srt_file: Path) -> list[dict]:
    """Преобразует SRT в JSON, сохраняя все блоки и переносы строк."""
    with srt_file.open("r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
    json_blocks = []
    for sub in subtitles:
        json_blocks.append({
            "index": sub.index,
            "start": str(sub.start),
            "end": str(sub.end),
            "lines": [{"text": line} for line in sub.content.splitlines()]
        })
    return json_blocks

def json_to_srt(json_blocks: list[dict]) -> str:
    """Преобразует JSON обратно в SRT."""
    subs = []
    for block in json_blocks:
        content = "\n".join([line["text"] for line in block["lines"]])
        start = timedelta(
            hours=block["start"].hour,
            minutes=block["start"].minute,
            seconds=block["start"].second,
            microseconds=block["start"].microsecond
        ) if isinstance(block["start"], timedelta) else srt.srt_timestamp_to_timedelta(block["start"])
        end = timedelta(
            hours=block["end"].hour,
            minutes=block["end"].minute,
            seconds=block["end"].second,
            microseconds=block["end"].microsecond
        ) if isinstance(block["end"], timedelta) else srt.srt_timestamp_to_timedelta(block["end"])
        subs.append(srt.Subtitle(index=block["index"], start=start, end=end, content=content))
    return srt.compose(subs)

def validate_json(json_blocks: list[dict]) -> list[dict]:
    """Проверка JSON через Pydantic."""
    validated = []
    for block in json_blocks:
        try:
            b = SubtitleBlock(
                index=block["index"],
                start=block["start"],
                end=block["end"],
                lines=[SubtitleLine(**line) for line in block["lines"]]
            )
            validated.append(b.dict())
        except ValidationError as e:
            print(f"Ошибка в блоке {block.get('index')}: {e}")
    return validated

# ---------- Основной скрипт ----------
def process_srt(input_srt: Path, output_srt: Path):
    # 1. SRT -> JSON
    json_blocks = srt_to_json(input_srt)
    total_blocks = len(json_blocks)
    print(f"Всего блоков для обработки: {total_blocks}")

    BATCH_SIZE = 10
    all_edited_blocks = []

    # 3. Обработка батчами
    for i in range(0, total_blocks, BATCH_SIZE):
        batch = json_blocks[i : i + BATCH_SIZE]
        print(f"Обработка батча {i // BATCH_SIZE + 1} из {(total_blocks + BATCH_SIZE - 1) // BATCH_SIZE} (блоки {i+1}-{min(i+BATCH_SIZE, total_blocks)})...")

        prompt = {
            "role": "user",
            "content": json.dumps(batch, ensure_ascii=False)
        }

        try:
            # 4. Вызов локальной модели
            response = ollama.chat(
                model="gemma3-srt-voice-over-editor-so:4b",
                messages=[prompt],
                format='json'
            )

            # 5. Получение JSON из ответа модели
            response_content = response.message.content
            # print(f"DEBUG: RAW response for batch: {response_content[:100]}...")
            
            edited_batch = json.loads(response_content)

            # Обработка возможного "одиночного" ответа или обернутого в dict
            if isinstance(edited_batch, dict):
                # Если модель вернула один объект вместо списка или обертку
                if "subtitles" in edited_batch and isinstance(edited_batch["subtitles"], list):
                     edited_batch = edited_batch["subtitles"]
                elif "blocks" in edited_batch and isinstance(edited_batch["blocks"], list):
                     edited_batch = edited_batch["blocks"]
                else: 
                     # Пытаемся понять, это один блок или что-то еще
                     # Если это один блок (есть index), обернем в список
                     if "index" in edited_batch:
                         edited_batch = [edited_batch]
                     else:
                        print(f"WARN: Непонятная структура ответа батча: {type(edited_batch)}")
                        # В худшем случае пропускаем или пробуем добавить как есть (упадет на валидации)

            if not isinstance(edited_batch, list):
                 print(f"WARN: Ответ модели не список, пропускаем батч. Тип: {type(edited_batch)}")
                 # Можно добавить логику retry, но пока просто пропустим или добавим оригинал?
                 # Для надежности лучше добавить оригинал, чтобы не сбился тайминг, но пометим ошибку.
                 print("WARN: Используем оригинальный батч из-за ошибки модели.")
                 all_edited_blocks.extend(batch)
                 continue

            # 6. Валидация ответа модели (батча)
            # Важно: модель может вернуть меньше блоков или испортить индексы. 
            # Здесь мы пока просто валидируем структуру.
            validated_batch = validate_json(edited_batch)
            all_edited_blocks.extend(validated_batch)

        except Exception as e:
            print(f"ERROR: Ошибка при обработке батча: {e}")
            print(f"DEBUG: Сырой ответ был: {response.message.content if 'response' in locals() else 'No response'}")
            # Fallback: добавляем оригинальные блоки, чтобы не ломать файл
            print("WARN: Используем оригинальный батч из-за ошибки.")
            all_edited_blocks.extend(batch)

    # 7. JSON -> SRT
    print(f"Сборка финального файла из {len(all_edited_blocks)} блоков...")
    
    # Дополнительная защита: сортируем по индексу, чтобы порядок был верным
    # (хотя append должен сохранить порядок)
    all_edited_blocks.sort(key=lambda x: x['index'])

    srt_text = json_to_srt(all_edited_blocks)

    # 8. Сохранение в файл
    with output_srt.open("w", encoding="utf-8") as f:
        f.write(srt_text)

    print(f"Готово! Отредактированные субтитры сохранены в {output_srt}")

# ---------- Запуск ----------
if __name__ == "__main__":
    input_file = Path("input.srt")
    output_file = Path("output_voiceover.srt")
    process_srt(input_file, output_file)