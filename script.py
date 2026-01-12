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

        # Подготовка списка текстов (flattening)
        # Собираем весь текст блока в одну строку (с переносами, если были), чтобы модель видела контекст фразы целиком
        batch_texts = []
        for block in batch:
            full_text = "\n".join(line["text"] for line in block["lines"])
            batch_texts.append(full_text)

        prompt = {
            "role": "user",
            "content": json.dumps(batch_texts, ensure_ascii=False)
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
            
            # 5. Получение JSON из ответа модели
            response_content = response.message.content
            
            try:
                response_data = json.loads(response_content)
                
                # Ожидаем структуру {"lines": ["...", "..."]}
                if isinstance(response_data, dict) and "lines" in response_data and isinstance(response_data["lines"], list):
                    edited_texts = response_data["lines"]
                elif isinstance(response_data, list):
                    # На случай, если модель все-таки вернет просто список (игнорируя инструкцию, так бывает)
                    print("INFO: Модель вернула список напрямую, принимаем.")
                    edited_texts = response_data
                else:
                    # Пытаемся найти хоть какой-то список
                    found_list = None
                    if isinstance(response_data, dict):
                        for k, v in response_data.items():
                            if isinstance(v, list):
                                found_list = v
                                print(f"WARN: Найден список по ключу '{k}', используем его.")
                                break
                    
                    if found_list:
                        edited_texts = found_list
                    else:
                        print(f"ERROR: Некорректная структура ответа. Получено: {type(response_data)}")
                        if isinstance(response_data, dict):
                            print(f"DEBUG: Ключи: {list(response_data.keys())}")
                        raise ValueError("Структура ответа не соответствует ожидаемой")

            except json.JSONDecodeError:
                print(f"ERROR: Ошибка парсинга JSON. Сырой ответ: {response_content[:100]}...")
                all_edited_blocks.extend(batch)
                continue
            except Exception as e:
                print(f"WARN: Ошибка обработки данных ({e}). Используем оригиналы.")
                all_edited_blocks.extend(batch)
                continue
            
            # Проверка длины (критично!)
            if len(edited_texts) != len(batch):
                print(f"WARN: Несовпадение длины! Отправлено {len(batch)}, получено {len(edited_texts)}. Используем оригиналы.")
                all_edited_blocks.extend(batch)
                continue

            # 6. Обновление блоков новыми текстами
            for block, new_text in zip(batch, edited_texts):
                # Обновляем текст. Если он пришел строкой, запишем его как одну линию или разобьем?
                # Для SRT проще оставить как одну линию с \n, srt библиотека разберется (или мы уже разбили в srt_to_json)
                # srt_to_json разбивает splitlines().
                # Мы можем просто заменить lines на один элемент с новым текстом (модель могла убрать переносы)
                if isinstance(new_text, str):
                    block["lines"] = [{"text": new_text}]
                else:
                     # Если вдруг модель вернула не строку (бред, но всё же)
                     block["lines"] = [{"text": str(new_text)}]
            
            all_edited_blocks.extend(batch)

        except Exception as e:
            print(f"ERROR: Ошибка при обработке батча: {e}")
            print("WARN: Используем оригинальный батч из-за ошибки.")
            all_edited_blocks.extend(batch)

    # 7. JSON -> SRT
    print(f"Сборка финального файла из {len(all_edited_blocks)} блоков...")
    
    # Дополнительная защита: сортируем по индексу
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