import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydub import AudioSegment
import openai
from openai import OpenAI

from dotenv import load_dotenv
from docx import Document

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_KEY"), base_url="https://api.deepseek.com")

app = FastAPI()
# Используем middleware для управления сессиями
app.add_middleware(SessionMiddleware, secret_key="!secret")

# Определяем директории для хранения загруженных файлов и результатов
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

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
    return HTMLResponse(content=html_content)

# Эндпоинт загрузки аудиофайла. Можно передать дополнительный параметр "prompt"
@app.post("/upload/")
async def upload_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(None)
):
    user_id = request.session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        request.session["user_id"] = user_id

    # Сохраняем загруженный файл на сервер
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f_out:
        content = await file.read()
        f_out.write(content)

    # Если пользователь не задал промпт, используем значение по умолчанию
    if not prompt:
        prompt = "Создай краткое содержание в виде списка ключевых тезисов на основе текста пользователя"

    # Запускаем фоновую задачу для обработки аудио
    background_tasks.add_task(process_audio, file_path, file_id, user_id, prompt)

    return JSONResponse({"message": "Файл загружен, обработка запущена", "file_id": file_id})

def process_audio(file_path: str, file_id: str, user_id: str, prompt: str):
    try:
        # 1. Разбивка аудио на части, если длительность больше 10 минут
        parts = split_audio(file_path)
        transcripts = []
        for part in parts:
            transcript = transcribe_with_whisper(part)
            transcripts.append(transcript)
            # Если файл является временным (от разбиения), удаляем его после транскрипции
            if part != file_path:
                os.remove(part)
        full_text = " ".join(transcripts)

        # 2. Генерация резюме через ChatGPT API
        summary = generate_summary(full_text, prompt)

        # 3. Создание итогового Word?файла (.docx)

        result_filename = create_docx(summary, file_id)

        # 4. Сохранение результата в истории пользователя (директория RESULT_DIR/user_id)
        user_dir = os.path.join(RESULT_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        final_path = os.path.join(user_dir, result_filename)
        shutil.move(result_filename, final_path)

        # 5. Удаляем исходный аудиофайл, если он существует
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error processing file {file_id}: {str(e)}")

def split_audio(file_path: str):
    """
    Разбивает аудиофайл на части, если его длительность превышает 15 минут.
    Возвращает список путей к файлам (если длительность меньше – возвращает исходный файл).
    """
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    max_duration = 10 * 60 * 1000  # 10 минут в миллисекундах

    if duration_ms <= max_duration:
        return [file_path]

    parts = []
    num_parts = (duration_ms // max_duration) + (1 if duration_ms % max_duration else 0)
    base, ext = os.path.splitext(file_path)
    for i in range(num_parts):
        start_ms = i * max_duration
        end_ms = min((i + 1) * max_duration, duration_ms)
        chunk = audio[start_ms:end_ms]
        part_filename = f"{base}_part{i+1}{ext}"
        # Экспортируем часть; формат определяется по расширению (без точки)
        chunk.export(part_filename, format=ext[1:])
        parts.append(part_filename)
    return parts


def transcribe_with_whisper(audio_path: str) -> str:
    """
    Выполняет транскрипцию аудио через Whisper API,
    используя отдельный API‑ключ и базовый URL для Whisper.
    """
    try:
        whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")

        with open(audio_path, "rb") as audio_file:
            response = whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        return response
    except Exception as e:
        print(f"Error transcribing {audio_path}: {str(e)}")
        return ""

def generate_summary(text: str, prompt: str) -> str:
    """
    Генерирует резюме, отправляя полный текст в DeepSeek API с указанным системным промптом.
    """
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        response = client.chat.completions.create(model="deepseek-chat",  # модель для DeepSeek
        messages=messages,
        temperature=0.5,
        max_tokens=800)
        # Обращаемся к результату через словарный синтаксис
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Ошибка генерации резюме"

def create_docx(summary: str, file_id: str):
    """
    Создает .docx файл с итоговым резюме.
    """
    doc = Document()
    doc.add_heading("Резюме", level=1)
    doc.add_paragraph(summary)
    output_filename = f"{file_id}_summary.docx"
    doc.save(output_filename)
    return output_filename

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
    files = os.listdir(user_dir)
    return JSONResponse({"history": files, "user_id": user_id})

# При необходимости можно смонтировать статические файлы (например, для CSS или JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
