import os
import uuid
import shutil
import asyncio
import subprocess
import json

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from enum import Enum
from urllib.parse import quote

from dotenv import load_dotenv


# ---------------------------------------------------------
# Сначала определяем BASE_DIR и загружаем .env
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))


# ---------------------------------------------------------
# Настраиваем FFmpeg ДО импорта pydub
# ---------------------------------------------------------

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "").strip()

if FFMPEG_PATH:
    ffmpeg_bin = os.path.dirname(FFMPEG_PATH)

    os.environ["PATH"] = (
        ffmpeg_bin
        + os.pathsep
        + os.environ.get("PATH", "")
    )


# ---------------------------------------------------------
# Только теперь остальные библиотеки
# ---------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from pydub import AudioSegment
from openai import OpenAI
from docx import Document
from faster_whisper import WhisperModel


# Явно задаём ffmpeg для pydub
if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH

def get_llm_api_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip() or os.getenv("DEEPSEEK_KEY", "").strip()


def get_llm_base_url() -> str:
    custom = os.getenv("LLM_BASE_URL", "").strip()
    if custom:
        return custom.rstrip("/")
    if os.getenv("LLM_API_KEY", "").strip():
        return "https://vedai.by/api/v1"
    return "https://api.deepseek.com"


def get_llm_client() -> OpenAI:
    key = get_llm_api_key()
    if not key:
        raise ValueError(
            "Нет ключа для резюме: укажите LLM_API_KEY (агрегатор) или DEEPSEEK_KEY."
        )
    return OpenAI(api_key=key, base_url=get_llm_base_url())


def llm_supports_thinking() -> bool:
    return "deepseek.com" in get_llm_base_url()


DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "change-me-in-production"),
)

# Определяем директории для хранения загруженных файлов и результатов
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ThreadPoolExecutor для фоновых задач (чтобы не блокировать event loop)
executor = ThreadPoolExecutor(max_workers=2)

# Хранение статусов задач (в памяти + файл, чтобы пережить перезапуск сервера)
TASKS_FILE = os.path.join(BASE_DIR, "task_statuses.json")
task_statuses: Dict[str, Dict] = {}


def load_task_statuses() -> None:
    global task_statuses
    if not os.path.exists(TASKS_FILE):
        return
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            task_statuses = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Не удалось загрузить статусы задач: {e}")
        task_statuses = {}


def save_task_statuses() -> None:
    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(task_statuses, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Не удалось сохранить статусы задач: {e}")


def update_task_status(file_id: str, **kwargs) -> None:
    if file_id not in task_statuses:
        task_statuses[file_id] = {}
    task_statuses[file_id].update(kwargs)
    save_task_statuses()


load_task_statuses()

class TranscriptionEngine(str, Enum):
    WHISPER = "whisper"
    SPEECHKIT = "speechkit"
    GROQ = "groq"
    STT_API = "stt_api"

AUDIO_SPLIT_MS = 10 * 60 * 1000
SUMMARY_CHUNK_CHARS = 40000
# Описание процесса отправляем целиком: нарезка рвёт порядок шагов и развилки.
# Если транскрипт всё же длиннее (это 8–10 часов речи) — последовательное уточнение черновика.
WHOLE_TEXT_MAX_CHARS = 500_000
REFINE_OVERLAP_CHARS = 2500

PROMPT_SHARED_RULES = """
Транскрипция может содержать ошибки распознавания, обрывки и повторы. Исправляй только очевидные оговорки. Не выдумывай факты, решения, имена, цифры, суммы и сроки.
Если исполнитель, срок или сумма не названы явно — пиши «не назначен» / «не указан» / «неясно». Не догадывайся.
Не пересказывай приветствия, болтовню и повторы. Каждое поручение — отдельная строка, задачи разных людей не сливай.
Если в транскрипции есть таймкоды вида [мм:сс] или [чч:мм:сс], ставь их в таблицу задач.
Язык — русский. Без канцелярита.
"""

PROMPT_PRESETS = {
    "default": """Ты — секретарь совещания. Составь протокол, по которому человек, который не был на встрече, за одну-две минуты поймёт: зачем собрались, что решили и кто что делает.
"""
    + PROMPT_SHARED_RULES
    + """
Формат (Markdown):

## Тема
Одно-два предложения: зачем собрались и чем закончили.

## Решения
Пронумерованный список только явных договорённостей («решили», «согласовали», «утвердили», «оставляем как есть»). Если решений не было: «Явных решений не зафиксировано».

## Задачи
Таблица. Одна задача — одна строка. Не добавляй задачи, которых не было.

| Задача | Ответственный | Срок | Таймкод |
| --- | --- | --- | --- |

## Важное
Только цифры, ограничения и условия, без которых решения и задачи теряют смысл. Максимум 7 пунктов. Если такого нет — раздел не пиши.

## Открытые вопросы
Что не решили. Если таких нет — раздел не пиши.
""",
    "tasks": """Ты — секретарь. Из транскрипции извлеки ТОЛЬКО поручения. Никакого пересказа встречи.
"""
    + PROMPT_SHARED_RULES
    + """
Формат — только заголовок и таблица, больше ничего:

## Задачи

| Задача | Ответственный | Срок | Таймкод |
| --- | --- | --- | --- |

Правила таблицы:
- Одна договорённость «сделать X» = одна строка.
- Ответственный — ФИО или роль, как сказано. Если не сказано — «не назначен».
- Срок — дата или формулировка из речи. Если не сказан — «не указан».
- Таймкод — ближайшая метка из транскрипции. Если меток нет — «—».
- Если поручений не было, вместо таблицы напиши: «Поручений не зафиксировано».
""",
    "brief": """Ты — помощник руководителя. Составь короткую служебную записку по транскрипции: что решили, какие риски, кто что делает.
"""
    + PROMPT_SHARED_RULES
    + """
Пиши плотно, без воды. Объём — не больше одной страницы.

Формат (Markdown):

## Коротко
3–5 предложений: зачем встреча, чем закончили, что будет дальше.

## Решения
Маркированный список, только итог. Если решений нет: «Явных решений не зафиксировано».

## Поручения
| Кто | Что сделать | Срок |
| --- | --- | --- |

Если поручений нет — раздел не пиши.

## Риски и развилки
Только то, что прямо назвали как проблему, риск или нерешённый выбор. Максимум 5 пунктов. Если такого нет — раздел не пиши.
""",
    "process": """Ты — бизнес-аналитик процессного офиса. В транскрипции сотрудник или отдел рассказывает, как устроена их работа. Опиши процесс «как есть» последовательно, шаг за шагом, так, чтобы по твоему тексту аналитик мог без повторного прослушивания нарисовать BPMN-схему.

Правила:
- Транскрипция может содержать ошибки распознавания, обрывки и повторы. Исправляй только очевидные оговорки.
- Описывай только то, что сказано. Не выдумывай шаги, роли, системы, документы, сроки и условия. Чего не хватает — пиши «не указано».
- Участник может быть один (работа одного человека) или несколько (отдел, смежники, клиент). Если участник один — все шаги от его роли, но внешние стороны (клиент, другой отдел, система) называй отдельно.
- Один шаг — одно действие. Формулировка шага: кто — что делает (глагол) — с чем — какой результат.
- Называй роли, а не только имена: «Менеджер (Иван)». Если роль не ясна — «исполнитель не указан».
- Развилки описывай явно, с условием и куда идёт процесс в каждом случае.
- Если в транскрипции есть таймкоды вида [мм:сс] или [чч:мм:сс], ставь ближайший в конце шага.
- Не пересказывай приветствия, отступления и повторы. Если рассказчик описывает процесс непоследовательно — восстанови порядок выполнения.
- Без таблиц и без жирного шрифта. Только заголовки, абзацы и списки.
- Язык — русский. Без канцелярита.

Формат (Markdown):

## Процесс
Название процесса и одно-два предложения: зачем он нужен и чей он (отдел или сотрудник).

## Границы
- Старт: какое событие запускает процесс (заявка, письмо, дата, звонок и т. п.).
- Финиш: чем процесс заканчивается и какой результат получается.

## Участники
- Роли и отделы внутри процесса.
- Внешние стороны (клиент, смежные отделы, подрядчики).
- Системы и документы, которые используются.

## Ход процесса
Нумерованные шаги в порядке выполнения, каждый с новой строки:
1. Роль — действие — с чем — результат. [таймкод]

Развилка — отдельным шагом с вариантами списком:
5. Роль проверяет условие: …
- Если …, то переход к шагу 6.
- Иначе — переход к шагу 9.

Параллельные работы помечай: «Одновременно: …». Ожидания и сроки указывай в шаге: «ждёт ответа до 3 рабочих дней». Последний шаг — конечное событие процесса.

## Исключения и альтернативные пути
Ошибки, возвраты на доработку, эскалации, отмены — где возникают (номер шага) и чем заканчиваются. Если таких нет — раздел не пиши.

## Пробелы в описании
Вопросы, которые нужно уточнить у рассказчика, чтобы построить корректную схему: неясные исполнители, условия развилок, что происходит при отказе, где хранится результат. Только по делу, максимум 10 пунктов.
""",
}

DEFAULT_SUMMARY_PROMPT = PROMPT_PRESETS["default"]


def resolve_summary_prompt(preset: str | None, custom_prompt: str | None) -> str:
    custom = (custom_prompt or "").strip()
    if custom:
        return custom
    key = (preset or "default").strip().lower()
    return PROMPT_PRESETS.get(key, DEFAULT_SUMMARY_PROMPT)


def resolve_summary_model(preset: str | None) -> tuple[str, bool, bool]:
    """Модель, режим рассуждения и отправка транскрипта целиком — для шаблона."""
    key = (preset or "default").strip().lower()
    aggregator = bool(os.getenv("LLM_API_KEY", "").strip())
    default_flash = "deepseek-v4-flash" if aggregator else "deepseek-flash"
    if key == "process":
        model = (
            os.getenv("LLM_PROCESS_MODEL", "").strip()
            or os.getenv("DEEPSEEK_PROCESS_MODEL", "").strip()
            or "deepseek-v4-pro"
        )
        thinking = os.getenv("DEEPSEEK_PROCESS_THINKING", "true").strip().lower() == "true"
        return model, thinking, True
    model = (
        os.getenv("LLM_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or default_flash
    )
    return model, False, False


def format_timestamp(seconds: float) -> str:
    total = int(max(0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def segments_to_transcript(segments, offset_sec: float = 0.0) -> tuple[str, str]:
    plain_parts = []
    timed_lines = []
    for segment in segments:
        if isinstance(segment, dict):
            text = (segment.get("text") or "").strip()
            start = float(segment.get("start") or 0)
        else:
            text = (getattr(segment, "text", "") or "").strip()
            start = float(getattr(segment, "start", 0) or 0)
        if not text:
            continue
        plain_parts.append(text)
        timed_lines.append(f"[{format_timestamp(start + offset_sec)}] {text}")
    return " ".join(plain_parts).strip(), "\n".join(timed_lines).strip()

# Кэш для модели Whisper (загружается один раз)
_whisper_model: Optional[WhisperModel] = None

def get_whisper_model() -> WhisperModel:
    """Получает или создает модель Whisper (кэшируется)"""
    global _whisper_model
    if _whisper_model is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        num_threads = int(os.getenv("WHISPER_NUM_THREADS", "4"))
        _whisper_model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            num_workers=num_threads
        )
    return _whisper_model

def check_ffmpeg():
    """Проверяет наличие ffmpeg в системе"""
    ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Пробуем найти в стандартных местах
    if os.path.exists(ffmpeg_path):
        return True
    
    raise RuntimeError(
        f"FFmpeg не найден. Установите FFmpeg и добавьте его в PATH, "
        f"или укажите путь через переменную окружения FFMPEG_PATH. "
        f"Текущий путь: {ffmpeg_path}"
    )

# Проверяем ffmpeg при старте приложения
try:
    check_ffmpeg()
except RuntimeError as e:
    print(f"Предупреждение: {e}")

# Маршрут для главной страницы (фронтенд)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Если в сессии отсутствует user_id, генерируем новый
    user_id = request.session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        request.session["user_id"] = user_id
    # Читаем файл index.html (находится в той же папке)
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html")


@app.get("/config")
async def public_config():
    """Публичные настройки для фронтенда: Google Picker и доступные движки."""
    return JSONResponse({
        "google_client_id": os.getenv("GOOGLE_CLIENT_ID", "").strip(),
        "google_api_key": os.getenv("GOOGLE_API_KEY", "").strip(),
        "google_app_id": os.getenv("GOOGLE_APP_ID", "").strip(),
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
    })


# Эндпоинт загрузки аудиофайла. Можно передать дополнительный параметр "prompt" и "engine"
@app.post("/upload/")
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(None),
    preset: str = Form("default"),
    engine: str = Form("whisper")
):
    user_id = request.session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        request.session["user_id"] = user_id

    # Валидация движка
    try:
        engine_enum = TranscriptionEngine(engine.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный движок: {engine}. Доступны: whisper, speechkit, groq, stt_api",
        )

    # Сохраняем загруженный файл на сервер
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f_out:
        content = await file.read()
        f_out.write(content)

    prompt = resolve_summary_prompt(preset, prompt)
    model, thinking, whole_text = resolve_summary_model(preset)

    # Инициализируем статус задачи
    update_task_status(
        file_id,
        status="queued",
        progress=0,
        message="Задача поставлена в очередь",
        download_url=None,
        transcript_url=None,
        error=None,
    )

    # Запускаем фоновую задачу для обработки аудио через executor
    loop = asyncio.get_event_loop()
    original_name = file.filename or "audio"
    loop.run_in_executor(
        executor,
        process_audio,
        file_path,
        file_id,
        user_id,
        prompt,
        engine_enum,
        original_name,
        model,
        thinking,
        whole_text,
    )

    return JSONResponse({
        "message": "Файл загружен, обработка запущена",
        "file_id": file_id
    })

# Эндпоинт для получения статуса задачи
@app.get("/status/{file_id}")
async def get_status(file_id: str):
    if file_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JSONResponse(task_statuses[file_id])

def process_audio(
    file_path: str,
    file_id: str,
    user_id: str,
    prompt: str,
    engine: TranscriptionEngine,
    original_name: str = "audio",
    model: str = DEEPSEEK_MODEL,
    thinking: bool = False,
    whole_text: bool = False,
):
    try:
        # Обновляем статус: начата обработка
        update_task_status(
            file_id,
            status="running",
            progress=10,
            message="Начата обработка аудио",
        )

        # 1. Разбивка аудио на части, если длительность больше 10 минут
        update_task_status(
            file_id,
            progress=20,
            message="Разбивка аудио на части...",
        )
        parts = split_audio(file_path)
        
        # 2. Транскрибация
        update_task_status(
            file_id,
            progress=30,
            message=f"Транскрибация через {engine.value}...",
        )
        transcripts = []
        timed_transcripts = []
        total_parts = len(parts)
        for idx, part in enumerate(parts):
            offset_sec = idx * (AUDIO_SPLIT_MS / 1000)
            plain, timed = transcribe_audio(part, engine, offset_sec)
            transcripts.append(plain)
            timed_transcripts.append(timed)
            # Если файл является временным (от разбиения), удаляем его после транскрипции
            if part != file_path:
                os.remove(part)
            
            # Обновляем прогресс
            progress = 30 + int((idx + 1) / total_parts * 50)
            update_task_status(
                file_id,
                progress=progress,
                message=f"Транскрибация: {idx + 1}/{total_parts} частей",
            )
        
        full_text = " ".join(t for t in transcripts if t).strip()
        timed_text = "\n".join(t for t in timed_transcripts if t).strip() or full_text

        # Проверяем, что транскрипция не пуста
        if not full_text:
            raise ValueError("Транскрипция пуста. Возможно, аудио не содержит речи или произошла ошибка распознавания.")

        # 3. Генерация резюме через DeepSeek API
        update_task_status(file_id, progress=85, message=f"Генерация резюме ({model})...")
        summary = generate_summary(full_text, prompt, model, thinking, whole_text)

        # 4. Два отдельных Word-файла: резюме и транскрипт
        update_task_status(file_id, progress=90, message="Создание документов...")
        summary_name, transcript_name = create_result_documents(
            summary, timed_text, original_name, file_id
        )

        user_dir = os.path.join(RESULT_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        _move_result_file(summary_name, user_dir)
        _move_result_file(transcript_name, user_dir)

        if os.path.exists(file_path):
            os.remove(file_path)

        update_task_status(
            file_id,
            status="done",
            progress=100,
            message="Обработка завершена",
            download_url=f"/download/{user_id}/{quote(summary_name)}",
            transcript_url=f"/download/{user_id}/{quote(transcript_name)}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing file {file_id}: {error_msg}")
        update_task_status(
            file_id,
            status="error",
            message=f"Ошибка обработки: {error_msg}",
            error=error_msg,
        )

def prepare_audio_for_transcription(audio_path: str) -> str:
    """
    Подготавливает аудио для транскрибации: конвертирует в моно, 16kHz, WAV.
    Возвращает путь к подготовленному файлу.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        # Конвертируем в моно и 16kHz для оптимизации
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Сохраняем во временный WAV файл
        base, _ = os.path.splitext(audio_path)
        prepared_path = f"{base}_prepared.wav"
        audio.export(prepared_path, format="wav")
        return prepared_path
    except Exception as e:
        print(f"Ошибка подготовки аудио {audio_path}: {e}")
        # Если не удалось подготовить, возвращаем исходный файл
        return audio_path

def split_audio(file_path: str):
    """
    Разбивает аудиофайл на части, если его длительность превышает 10 минут.
    Возвращает список путей к файлам (если длительность меньше – возвращает исходный файл).
    """
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить аудиофайл {file_path}. "
            f"Убедитесь, что файл не поврежден и FFmpeg установлен корректно. Ошибка: {str(e)}"
        )
    
    duration_ms = len(audio)
    max_duration = AUDIO_SPLIT_MS

    if duration_ms <= max_duration:
        return [file_path]

    parts = []
    num_parts = (duration_ms // max_duration) + (1 if duration_ms % max_duration else 0)
    base, _ = os.path.splitext(file_path)
    for i in range(num_parts):
        try:
            start_ms = i * max_duration
            end_ms = min((i + 1) * max_duration, duration_ms)
            chunk = audio[start_ms:end_ms]
            # Временные части всегда в WAV — надёжнее, чем повторный экспорт в m4a/mp3
            part_filename = f"{base}_part{i+1}.wav"
            chunk.export(part_filename, format="wav")
            parts.append(part_filename)
        except Exception as e:
            # Очищаем уже созданные части при ошибке
            for part in parts:
                try:
                    if os.path.exists(part):
                        os.remove(part)
                except:
                    pass
            raise RuntimeError(f"Ошибка при разбиении аудио на части: {str(e)}")
    
    return parts


def transcribe_audio(audio_path: str, engine: TranscriptionEngine, offset_sec: float = 0.0) -> tuple[str, str]:
    if engine == TranscriptionEngine.WHISPER:
        return transcribe_with_whisper(audio_path, offset_sec)
    if engine == TranscriptionEngine.SPEECHKIT:
        text = transcribe_with_speechkit(audio_path)
        return text, text
    if engine == TranscriptionEngine.GROQ:
        return transcribe_with_openai_stt(audio_path, "groq", offset_sec)
    if engine == TranscriptionEngine.STT_API:
        return transcribe_with_openai_stt(audio_path, "stt_api", offset_sec)
    raise ValueError(f"Неизвестный движок: {engine}")


def transcribe_with_whisper(audio_path: str, offset_sec: float = 0.0) -> tuple[str, str]:
    """
    Выполняет транскрипцию аудио через локальный faster-whisper (CPU).
    Видеокарта не нужна.
    """
    try:
        prepared_path = prepare_audio_for_transcription(audio_path)
        is_prepared = (prepared_path != audio_path)

        try:
            model = get_whisper_model()
            language = os.getenv("WHISPER_LANGUAGE", "ru") or None
            segments, info = model.transcribe(
                prepared_path,
                beam_size=5,
                language=language,
                task="transcribe",
            )
            plain, timed = segments_to_transcript(segments, offset_sec)
            if is_prepared and os.path.exists(prepared_path):
                os.remove(prepared_path)
            return plain, timed
        except Exception:
            if is_prepared and os.path.exists(prepared_path):
                try:
                    os.remove(prepared_path)
                except OSError:
                    pass
            raise
    except Exception as e:
        print(f"Error transcribing {audio_path} with Whisper: {str(e)}")
        raise


def transcribe_with_openai_stt(audio_path: str, provider: str, offset_sec: float = 0.0) -> tuple[str, str]:
    """
    Транскрипция через OpenAI-совместимый Speech-to-Text API.
    Groq: whisper-large-v3-turbo, около $0.04 за час аудио.
    """
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        base_url = os.getenv("GROQ_STT_BASE_URL", "https://api.groq.com/openai/v1").strip()
        model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip()
        if not api_key:
            raise ValueError("Для Groq нужен GROQ_API_KEY. Получить ключ: https://console.groq.com/keys")
    else:
        api_key = (
            os.getenv("STT_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        base_url = os.getenv("STT_API_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.getenv("STT_API_MODEL", "whisper-1").strip()
        if not api_key:
            raise ValueError(
                "Для внешнего транскрибатора нужен STT_API_KEY или OPENAI_API_KEY."
            )

    language = os.getenv("STT_LANGUAGE", os.getenv("WHISPER_LANGUAGE", "ru")).strip() or None
    stt_client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))

    upload_path = audio_path
    temp_mp3 = None
    try:
        audio = AudioSegment.from_file(audio_path)
        temp_mp3 = os.path.join(UPLOAD_DIR, f"stt_{uuid.uuid4()}.mp3")
        audio.export(temp_mp3, format="mp3", bitrate="64k")
        upload_path = temp_mp3
    except Exception as e:
        print(f"Не удалось сжать аудио для STT API, отправляем исходный файл: {e}")

    def _transcribe_file(response_format: str):
        with open(upload_path, "rb") as audio_file:
            kwargs = {
                "model": model,
                "file": audio_file,
                "response_format": response_format,
            }
            if language:
                kwargs["language"] = language
            return stt_client.audio.transcriptions.create(**kwargs)

    try:
        try:
            result = _transcribe_file("verbose_json")
        except Exception:
            result = _transcribe_file("json")

        if isinstance(result, str):
            text = result.strip()
            return text, text

        text = (getattr(result, "text", None) or "").strip()
        raw_segments = getattr(result, "segments", None)
        if raw_segments is None and isinstance(result, dict):
            text = (result.get("text") or text).strip()
            raw_segments = result.get("segments")

        if raw_segments:
            plain, timed = segments_to_transcript(raw_segments, offset_sec)
            return plain or text, timed or text
        return text, text
    except Exception as e:
        print(f"Error transcribing {audio_path} with {provider} STT: {str(e)}")
        raise
    finally:
        if temp_mp3 and os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except OSError:
                pass

def transcribe_with_speechkit(audio_path: str) -> str:
    """
    Выполняет транскрипцию аудио через Yandex SpeechKit REST API.
    Для длинных аудио разбивает на короткие фрагменты (20-30 секунд).
    """
    try:
        api_key = os.getenv("YANDEX_API_KEY", "").strip()
        folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        lang = os.getenv("YANDEX_LANG", "ru-RU")
        
        if not api_key or not folder_id:
            raise ValueError(
                "Yandex SpeechKit требует YANDEX_API_KEY и YANDEX_FOLDER_ID. "
                "Установите их в переменных окружения."
            )
        
        # Загружаем аудио
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        # SpeechKit REST API имеет ограничения, поэтому разбиваем на фрагменты по 25 секунд
        chunk_duration_ms = 25 * 1000  # 25 секунд
        
        if duration_ms <= chunk_duration_ms:
            # Короткое аудио - обрабатываем целиком
            return _transcribe_speechkit_chunk(audio, api_key, folder_id, lang)
        else:
            # Длинное аудио - разбиваем на части
            transcripts = []
            num_chunks = (duration_ms // chunk_duration_ms) + (1 if duration_ms % chunk_duration_ms else 0)
            
            for i in range(num_chunks):
                start_ms = i * chunk_duration_ms
                end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Транскрибируем чанк
                chunk_text = _transcribe_speechkit_chunk(chunk, api_key, folder_id, lang)
                if chunk_text:
                    transcripts.append(chunk_text)
            
            return " ".join(transcripts)
    except Exception as e:
        print(f"Error transcribing {audio_path} with SpeechKit: {str(e)}")
        raise

def _transcribe_speechkit_chunk(audio_segment: AudioSegment, api_key: str, folder_id: str, lang: str) -> str:
    """
    Транскрибирует один фрагмент аудио через SpeechKit REST API.
    """
    import requests
    
    # Конвертируем в OGG Opus (формат, поддерживаемый SpeechKit)
    # Создаем временный файл
    temp_file = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.ogg")
    try:
        # Экспортируем в OGG Opus
        audio_segment.export(temp_file, format="ogg", codec="libopus", parameters=["-ar", "16000"])
        
        # Читаем файл как бинарные данные
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        
        # Формируем запрос к SpeechKit
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        headers = {
            "Authorization": f"Api-Key {api_key}"
        }
        
        # Параметры в query string
        params = {
            "folderId": folder_id,
            "format": "oggopus",
            "lang": lang,
            "sampleRateHertz": "16000",
            "topic": "general"  # Можно настроить через env
        }
        
        # Отправляем запрос: бинарные данные в теле, параметры в query
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=audio_data
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise RuntimeError(f"SpeechKit API вернул ошибку {response.status_code}: {error_text}")
        
        result = response.json()

        # SpeechKit REST v1: {"result": "распознанный текст"}
        if isinstance(result.get("result"), str):
            return result["result"].strip()

        # Формат v2/v3 с alternatives (на случай смены API)
        nested = result.get("result")
        if isinstance(nested, dict):
            alternatives = nested.get("alternatives", [])
            if alternatives:
                return alternatives[0].get("text", "").strip()

        print(f"SpeechKit: неожиданный формат ответа: {result}")
        return ""
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

THINKING_TOKEN_BUDGET = 32000


def _chat_completion(
    system_prompt: str,
    user_text: str,
    max_tokens: int = 4000,
    model: str = DEEPSEEK_MODEL,
    thinking: bool = False,
) -> str:
    get_llm_api_key()
    params = {}
    extra_body = None
    if thinking:
        max_tokens += THINKING_TOKEN_BUDGET
    else:
        params["temperature"] = 0.4
    if llm_supports_thinking():
        extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        **params,
    }
    if extra_body is not None:
        request["extra_body"] = extra_body
    response = get_llm_client().chat.completions.create(**request)
    message = response.choices[0].message
    text = (getattr(message, "content", None) or "").strip()
    if not text and not thinking:
        text = (getattr(message, "reasoning_content", None) or "").strip()
    if not text:
        raise RuntimeError(
            f"Модель {model} вернула пустой ответ. Проверьте LLM_API_KEY / DEEPSEEK_KEY и имя модели."
        )
    return text


def _split_text_chunks(text: str, max_chars: int) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _refine_summary(text: str, prompt: str, llm: dict) -> str:
    """Черновик по первой части, затем каждая следующая часть дополняет его.
    К части добавляем хвост предыдущей, чтобы не терять шаги на стыке."""
    chunks = _split_text_chunks(text, WHOLE_TEXT_MAX_CHARS)
    total = len(chunks)
    draft = ""
    prev_tail = ""
    for index, chunk in enumerate(chunks, 1):
        if index == 1:
            user_text = (
                f"Это часть 1/{total} одной записи. Составь по ней черновик документа "
                f"строго в формате системного промпта.\n\n{chunk}"
            )
        else:
            user_text = (
                f"Ниже черновик документа по частям 1–{index - 1} одной записи, конец предыдущей части "
                f"(для связки) и часть {index}/{total}. Дополни и исправь черновик с учётом новой части: "
                "вставляй шаги на правильное место, сохраняй сквозную нумерацию и ссылки развилок, "
                "ничего из черновика не теряй без причины, не выдумывай. Верни полный обновлённый документ "
                "строго в формате системного промпта.\n\n"
                f"=== Черновик ===\n{draft}\n\n"
                f"=== Конец предыдущей части ===\n{prev_tail}\n\n"
                f"=== Часть {index}/{total} ===\n{chunk}"
            )
        draft = _chat_completion(prompt, user_text, max_tokens=8000, **llm)
        prev_tail = chunk[-REFINE_OVERLAP_CHARS:]
    return draft


def generate_summary(
    text: str,
    prompt: str,
    model: str = DEEPSEEK_MODEL,
    thinking: bool = False,
    whole_text: bool = False,
) -> str:
    """
    Генерирует резюме через DeepSeek. Длинные транскрипты режет на части,
    затем собирает одно резюме — иначе модель теряет решения и задачи.
    whole_text: отправить транскрипт целиком (до WHOLE_TEXT_MAX_CHARS), а сверх этого —
    последовательно уточнять один черновик, а не собирать его из независимых выжимок.
    """
    llm = {"model": model, "thinking": thinking}
    try:
        if whole_text:
            if len(text) <= WHOLE_TEXT_MAX_CHARS:
                return _chat_completion(prompt, text, max_tokens=8000, **llm)
            return _refine_summary(text, prompt, llm)
        if len(text) <= SUMMARY_CHUNK_CHARS:
            return _chat_completion(prompt, text, **llm)

        chunks = _split_text_chunks(text, SUMMARY_CHUNK_CHARS)
        partials = []
        for index, chunk in enumerate(chunks, 1):
            partials.append(
                _chat_completion(
                    prompt,
                    f"Это фрагмент {index}/{len(chunks)} одной записи. Извлеки из него материал "
                    f"строго по правилам системного промпта. Если в фрагменте нечего извлекать — напиши «пусто».\n\n{chunk}",
                    max_tokens=2500,
                    **llm,
                )
            )
        merged = "\n\n".join(
            f"## Фрагмент {index}\n{part}" for index, part in enumerate(partials, 1)
        )
        return _chat_completion(
            prompt,
            "Ниже извлечения из фрагментов одной записи. Собери из них ОДИН документ "
            "строго в формате системного промпта. Без повторов, без выдуманных фактов, "
            "действия и задачи разных участников не сливай:\n\n"
            + merged,
            **llm,
        )
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Ошибка генерации резюме через DeepSeek API: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    if not cells:
        return False
    return all(cell.replace(":", "").replace("-", "") == "" and "-" in cell for cell in cells)


def _add_markdown_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r_index, row in enumerate(rows):
        for c_index, value in enumerate(row):
            if c_index >= len(table.rows[r_index].cells):
                continue
            cell = table.rows[r_index].cells[c_index]
            cell.text = value
            if r_index == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
    doc.add_paragraph()


def _add_summary_blocks(doc: Document, summary: str) -> None:
    lines = [raw.rstrip() for raw in summary.splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                current = lines[index].strip()
                if not _is_table_separator(current):
                    table_lines.append(_table_cells(current))
                index += 1
            width = max((len(row) for row in table_lines), default=0)
            normalized = [row + [""] * (width - len(row)) for row in table_lines]
            _add_markdown_table(doc, normalized)
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=2)
        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        else:
            doc.add_paragraph(line)
        index += 1


def _move_result_file(filename: str, user_dir: str) -> None:
    dest = os.path.join(user_dir, filename)
    abs_result_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(abs_result_path):
        shutil.move(abs_result_path, dest)
    elif os.path.exists(filename):
        shutil.move(filename, dest)


def _new_document(title: str, original_name: str) -> Document:
    from pathlib import Path

    doc = Document()
    doc.add_heading(title, level=1)
    source = Path(original_name).name
    if source:
        meta = doc.add_paragraph()
        meta.add_run("Источник: ").bold = True
        meta.add_run(source)
    return doc


def create_result_documents(
    summary: str,
    transcript: str,
    original_name: str,
    file_id: str,
) -> tuple[str, str]:
    """Два отдельных .docx: резюме и транскрипт с таймкодами."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    suffix = file_id[:8]
    summary_name = f"Резюме_встречи_{stamp}_{suffix}.docx"
    transcript_name = f"Транскрипт_встречи_{stamp}_{suffix}.docx"

    summary_doc = _new_document("Резюме встречи", original_name)
    _add_summary_blocks(summary_doc, summary)
    summary_doc.save(os.path.join(BASE_DIR, summary_name))

    transcript_doc = _new_document("Транскрипт встречи", original_name)
    for line in (transcript or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            transcript_doc.add_paragraph(cleaned)
    transcript_doc.save(os.path.join(BASE_DIR, transcript_name))

    return summary_name, transcript_name

# Эндпоинт для скачивания итогового файла
@app.get("/download/{user_id}/{filename}")
async def download_file(user_id: str, filename: str):
    file_path = os.path.join(RESULT_DIR, user_id, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="Файл не найден")

# Эндпоинт для получения истории обработанных файлов текущего пользователя
@app.get("/history", response_class=JSONResponse)
async def get_history(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"history": [], "user_id": None})
    user_dir = os.path.join(RESULT_DIR, user_id)
    if not os.path.exists(user_dir):
        return JSONResponse({"history": [], "user_id": user_id})
    files = []
    for name in sorted(os.listdir(user_dir), reverse=True):
        files.append({
            "name": name,
            "url": f"/download/{user_id}/{quote(name)}",
        })
    return JSONResponse({"history": files, "user_id": user_id})

# При необходимости можно смонтировать статические файлы (например, для CSS или JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", port=8000, reload=reload)
