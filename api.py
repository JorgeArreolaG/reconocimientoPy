from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os

app = FastAPI()

# Carpeta temporal para subir imágenes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Imagen conocida en el sistema
KNOWN_FACE = "images/persona1.jpg"

@app.post("/verify-face")
async def verify_face(file: UploadFile = File(...)):
    # Guardar la imagen subida
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Comparar la cara
        result = DeepFace.verify(KNOWN_FACE, file_path, enforce_detection=True)
        return JSONResponse({"verified": result["verified"], "distance": result["distance"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
