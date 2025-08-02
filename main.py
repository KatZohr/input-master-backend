import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from io import BytesIO

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Input Master API",
    description="Audio processing backend for Input Master app",
    version="1.2.0"
)

# --- Set API Key from Environment Variable ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# --- CORS Configuration ---
origins = [
    "https://d4159febd.base44.com",
    "https://app--input-master-8d9b0621.base44.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    transcription: str
    original_filename: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Input Master Backend is running!", "version": "1.2.0"}

@app.post("/upload-audio/", response_model=TranscriptionResponse)
async def upload_audio_and_transcribe(
    file: UploadFile = File(...),
    userId: str = Form(...),
    reportType: str = Form(...)
):
    """
    Accepts an audio file, transcribes it using OpenAI Whisper,
    and returns the transcription text.
    """
    if not client:
        raise HTTPException(status_code=500, detail="API Key for transcription service is not configured.")

    try:
        # Read the file content
        audio_data = await file.read()
        
        # Create a file-like object for the API
        audio_file = BytesIO(audio_data)
        audio_file.name = file.filename

        print(f"Transcribing {file.filename} for user {userId}...")

        # --- Call OpenAI Whisper API (NEW SYNTAX) ---
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        transcription_text = transcript.text
        print(f"Transcription successful: '{transcription_text[:50]}...'")

        return TranscriptionResponse(
            success=True,
            message=f"Successfully transcribed {file.filename}",
            transcription=transcription_text,
            original_filename=file.filename
        )

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/debug")
def debug_environment():
    """Debug endpoint to check environment variables"""
    return {
        "openai_key_exists": bool(os.getenv("OPENAI_API_KEY")),
        "openai_key_length": len(os.getenv("OPENAI_API_KEY", "")),
        "openai_key_prefix": os.getenv("OPENAI_API_KEY", "")[:10] if os.getenv("OPENAI_API_KEY") else "None",
    }



