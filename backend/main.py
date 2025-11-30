from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import shutil
import os
import tempfile
from backend.transcription import transcribe_audio

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock flag - can be set via env var or assumed True for this sandbox
USE_MOCK = os.getenv("USE_MOCK_TRANSCRIPTION", "True").lower() == "true"

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint to handle audio file upload and return MusicXML.
    """
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Process the file
            musicxml_content = transcribe_audio(tmp_path, use_mock=USE_MOCK)

            # Return XML with correct MIME type
            return Response(content=musicxml_content, media_type="application/xml")

        finally:
            # Cleanup uploaded file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "mock_mode": USE_MOCK}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
