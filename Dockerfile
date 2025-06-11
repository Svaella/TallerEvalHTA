FROM python:3.10-slim

# Instala dependencias del sistema necesarias para LightGBM y otros paquetes
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto
EXPOSE 8000

# Ejecuta la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
