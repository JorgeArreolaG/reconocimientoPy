from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os

app = FastAPI()

# 🔐 CORS (OBLIGATORIO para Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta temporal para subir imágenes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Carpeta base donde se guardan las fotos de empleados organizadas por empresa
BASE_IMAGES_DIR = "images"

# PRE-CARGAR EL MODELO AL INICIAR LA APLICACIÓN
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("🔄 Cargando modelos de DeepFace...")
    print("=" * 50)
    try:
        # Cargar el modelo haciendo una verificación dummy
        DeepFace.build_model("VGG-Face")
        print("✅ Modelos cargados correctamente")
        print("🚀 API lista para recibir peticiones")
    except Exception as e:
        print(f"⚠️ Error al pre-cargar modelo: {e}")
    print("=" * 50)

@app.get("/")
async def root():
    return {
        "message": "API de Verificación Facial",
        "status": "activo",
        "endpoints": ["/verify-face", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "API funcionando correctamente"
    }

@app.post("/verify-face")
async def verify_face(
    codigo_empresa: str = Form(None),
    codigo_empleado: str = Form(None),
    file: UploadFile = File(None)
):
    print(f"✅ Recibiendo petición:")
    print(f"   - Empresa: {codigo_empresa}")
    print(f"   - Empleado: {codigo_empleado}")
    print(f"   - Archivo: {file.filename if file else 'No file'}")
    print(f"   - Tipo: {file.content_type if file else 'N/A'}")
    
    # Validar que llegaron todos los parámetros
    if not codigo_empresa:
        return JSONResponse({"error": "Falta codigo_empresa"}, status_code=400)
    if not codigo_empleado:
        return JSONResponse({"error": "Falta codigo_empleado"}, status_code=400)
    if not file:
        return JSONResponse({"error": "Falta el archivo"}, status_code=400)
    
    # Construir la ruta de la imagen conocida
    known_face_path = os.path.join(
        BASE_IMAGES_DIR, 
        codigo_empresa, 
        f"{codigo_empleado}.jpg"
    )


    # Verificar que la imagen de referencia existe
    if not os.path.exists(known_face_path):
        print(f"❌ No se encontró: {known_face_path}")
        return JSONResponse(
            {
                "error": f"No se encontró la imagen de referencia para el empleado {codigo_empleado} en la empresa {codigo_empresa}",
                "verified": False
            },
            status_code=404
        )
    
    # Guardar la imagen subida temporalmente
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print("🔍 Comparando rostros...")
        # Comparar las caras
        result = DeepFace.verify(
            known_face_path, 
            file_path, 
            model_name="VGG-Face",
            enforce_detection=False
        )
        
        print(f"✅ Resultado: verified={result['verified']}, distance={result['distance']}")
        
        return JSONResponse({
            "verified": result["verified"],
            "distance": result["distance"],
            "codigo_empresa": codigo_empresa,
            "codigo_empleado": codigo_empleado
        })
    except Exception as e:
        print(f"❌ Error en verificación: {str(e)}")
        return JSONResponse(
            {
                "error": str(e),
                "verified": False
            }, 
            status_code=400
        )
    finally:
        # SIEMPRE eliminar la imagen temporal
        if os.path.exists(file_path):
            os.remove(file_path)
            print("🗑️ Imagen temporal eliminada")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)