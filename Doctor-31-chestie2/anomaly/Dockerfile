# Imagine de bază
FROM python:3.10-slim

# Setează directorul de lucru
WORKDIR /app

# Copiază fișierele în container
COPY . .

# Instalează dependințele
RUN pip install --no-cache-dir -r requirements.txt

# Portul pe care rulează Streamlit
EXPOSE 8501

# Comanda care pornește aplicația
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
