import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydub import AudioSegment
import openai
from docx import Document

# ������� ��� OpenAI API ���� (��� ����� ������ ����� ���������� ���������)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
# ���������� middleware ��� ���������� ��������
app.add_middleware(SessionMiddleware, secret_key="!secret")

# ���������� ���������� ��� �������� ����������� ������ � �����������
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ������� ��� ������� �������� (��������)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # ���� � ������ ����������� user_id, ���������� �����
    user_id = request.session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        request.session["user_id"] = user_id
    # ������ ���� index.html (��������� � ��� �� �����)
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# �������� �������� ����������. ����� �������� �������������� �������� "prompt"
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

    # ��������� ����������� ���� �� ������
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f_out:
        content = await file.read()
        f_out.write(content)

    # ���� ������������ �� ����� ������, ���������� �������� �� ���������
    if not prompt:
        prompt = "������ ������� ���������� � ���� ������ �������� ������� �� ������ ���������� ������:"

    # ��������� ������� ������ ��� ��������� �����
    background_tasks.add_task(process_audio, file_path, file_id, user_id, prompt)

    return JSONResponse({"message": "���� ��������, ��������� ��������", "file_id": file_id})

def process_audio(file_path: str, file_id: str, user_id: str, prompt: str):
    try:
        # 1. �������� ����� �� �����, ���� ������������ ������ 15 �����
        parts = split_audio(file_path)
        transcripts = []
        for part in parts:
            transcript = transcribe_with_whisper(part)
            transcripts.append(transcript)
            # ���� ���� �������� ��������� (�� ���������), ������� ��� ����� ������������
            if part != file_path:
                os.remove(part)
        full_text = " ".join(transcripts)

        # 2. ��������� ������ ����� ChatGPT API
        summary = generate_summary(full_text, prompt)

        # 3. �������� ��������� Word?����� (.docx)
        result_filename = create_docx(summary, file_id)

        # 4. ���������� ���������� � ������� ������������ (���������� RESULT_DIR/user_id)
        user_dir = os.path.join(RESULT_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        final_path = os.path.join(user_dir, result_filename)
        shutil.move(result_filename, final_path)
    except Exception as e:
        print(f"Error processing file {file_id}: {str(e)}")

def split_audio(file_path: str):
    """
    ��������� ��������� �� �����, ���� ��� ������������ ��������� 15 �����.
    ���������� ������ ����� � ������ (���� ������������ ������ � ���������� �������� ����).
    """
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    max_duration = 15 * 60 * 1000  # 15 ����� � �������������

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
        # ������������ �����; ������ ������������ �� ���������� (��� �����)
        chunk.export(part_filename, format=ext[1:])
        parts.append(part_filename)
    return parts

def transcribe_with_whisper(audio_path: str):
    """
    ��������� ������������ ����� ����� Whisper API.
    ��� �������� ���������� ������������ openai.Audio.transcribe (��� ������� ���������).
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        # ���� ������������ ������� � ������ "text"
        if isinstance(transcript, dict) and "text" in transcript:
            return transcript["text"]
        elif isinstance(transcript, str):
            return transcript
        else:
            return ""
    except Exception as e:
        print(f"Error transcribing {audio_path}: {str(e)}")
        return ""

def generate_summary(text: str, prompt: str):
    """
    ���������� ������, ��������� ������ ����� � ChatGPT API � ��������� ��������� ��������.
    """
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        summary = response.choices[0].message['content']
        return summary
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "������ ��������� ������"

def create_docx(summary: str, file_id: str):
    """
    ������� .docx ���� � �������� ������.
    """
    doc = Document()
    doc.add_heading("������", level=1)
    doc.add_paragraph(summary)
    output_filename = f"{file_id}_summary.docx"
    doc.save(output_filename)
    return output_filename

# �������� ��� ���������� ��������� �����
@app.get("/download/{user_id}/{filename}")
async def download_file(user_id: str, filename: str):
    file_path = os.path.join(RESULT_DIR, user_id, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="���� �� ������")

# �������� ��� ��������� ������� ������������ ������ �������� ������������
@app.get("/history", response_class=JSONResponse)
async def get_history(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"history": []})
    user_dir = os.path.join(RESULT_DIR, user_id)
    if not os.path.exists(user_dir):
        return JSONResponse({"history": []})
    files = os.listdir(user_dir)
    return JSONResponse({"history": files})

# ��� ������������� ����� ������������ ����������� ����� (��������, ��� CSS ��� JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
