import os
import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Input Master API",
    description="Audio processing backend for Input Master app",
    version="1.1.0"
)

# --- Set API Key from Environment Variable ---
# This safely reads the key you added in the Railway "Variables" tab.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set.")
    # In a real production app, you might want to raise an error here.
else:
    openai.api_key = api_key


# --- CORS Configuration ---
origins = [
    "https://d4159febd.base44.com",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Pydantic Models for Data Validation ---
class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    transcription: str
    original_filename: str

# --- API Endpoints ---
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
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="API Key for transcription service is not configured.")

    try:
        # Read the file content in-memory
        audio_data = await file.read()
        
        # Save the in-memory file to a temporary file-like object for the API
        from io import BytesIO
        audio_file = BytesIO(audio_data)
        audio_file.name = file.filename # The API needs a file name

        print(f"Transcribing {file.filename} for user {userId}...")

        # --- Call OpenAI Whisper API for Transcription ---
        transcript_object = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
        
        transcription_text = transcript_object['text']
        print(f"Transcription successful: '{transcription_text[:50]}...'")

        # In the future, you would take `transcription_text` and `reportType`
        # and pass them to another AI model to generate the final report.

        return TranscriptionResponse(
            success=True,
            message="File transcribed successfully.",
            transcription=transcription_text,
            original_filename=file.filename
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process audio file. Error: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Input Master Backend is running!", "transcription_service_configured": bool(openai.api_key)}


