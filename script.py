import json
from pathlib import Path
from pydantic import BaseModel, ValidationError, conint, constr
from datetime import timedelta
import srt
from ollama import Ollama

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

    # 2. Валидация JSON
    json_blocks = validate_json(json_blocks)

    # 3. Подготовка запроса для модели Ollama
    ollama = Ollama()
    prompt = {
        "role": "user",
        "content": json.dumps(json_blocks, ensure_ascii=False)
    }

    # 4. Вызов локальной модели
    response = ollama.chat(
        model="Gemma3-srt-voice-over-editor:4b",
        messages=[prompt],
        temperature=0.7
    )

    # 5. Получение JSON из ответа модели
    try:
        edited_json = json.loads(response["content"])
    except Exception as e:
        raise RuntimeError(f"Ошибка разбора JSON из модели: {e}")

    # 6. Валидация ответа модели
    edited_json = validate_json(edited_json)

    # 7. JSON -> SRT
    srt_text = json_to_srt(edited_json)

    # 8. Сохранение в файл
    with output_srt.open("w", encoding="utf-8") as f:
        f.write(srt_text)

    print(f"Готово! Отредактированные субтитры сохранены в {output_srt}")

# ---------- Запуск ----------
if __name__ == "__main__":
    input_file = Path("input.srt")
    output_file = Path("output_voiceover.srt")
    process_srt(input_file, output_file)