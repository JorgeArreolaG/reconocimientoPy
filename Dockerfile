# Usamos Python 3.9 o 3.10 porque DeepFace es muy estable aquí
FROM python:3.9-slim

# Instalar dependencias del sistema para OpenCV y DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app y las carpetas de imágenes
COPY . .

# Crear carpetas necesarias
RUN mkdir -p uploads images

# Exponer el puerto de tu app (8080 según tu código)
EXPOSE 8080

# Comando para ejecutar
CMD ["python", "face.py"]