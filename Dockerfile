# Usa una imagen base ligera con Python
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia primero los archivos de dependencias (mejor para el cache de Docker)
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos del proyecto
COPY . .

# Crea el directorio de modelos (por si no existe)
RUN mkdir -p modelos

# Expone el puerto de la API
EXPOSE 8000

# Comando para ejecutar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

