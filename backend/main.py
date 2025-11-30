from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import shutil
import os
import tempfile
import json
from backend.transcription import transcribe_audio, transcribe_audio_pipeline

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock flag
USE_MOCK = os.getenv("USE_MOCK_TRANSCRIPTION", "True").lower() == "true"

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    stereo_mode: bool = Form(False),
    start_offset: float = Form(0.0), # Added based on requirements
    max_duration: float = Form(None)
):
    """
    Endpoint to handle audio file upload and return MusicXML.
    Supports stereo_mode form field.
    """
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Run pipeline
            # We use the new pipeline logic
            # Note: valid API response format is XML ("retain present format")

            result = transcribe_audio_pipeline(tmp_path, stereo_mode=stereo_mode, use_mock=USE_MOCK)

            # TODO: How to return analysis_data.json if user requested "retain present format" (XML)?
            # We will stick to returning XML as the body.
            # Optionally, we could attach JSON in a header, but that might be too large.
            # Or we could return a multipart response, but that changes the format.
            # We follow "retain present format" strict interpretation: Body is XML.

            # We can log the analysis data for debugging
            # print(json.dumps(result.analysis_data.to_dict(), indent=2))

            return Response(content=result.musicxml, media_type="application/xml")

        finally:
            # Cleanup uploaded file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"API Error: {e}")
        # Return 500
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "mock_mode": USE_MOCK}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
