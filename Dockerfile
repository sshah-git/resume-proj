FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=100 --retries=10 --resume-retries=5 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu



EXPOSE 8000

CMD ["uvicorn", "aiCheck:app", "--host", "0.0.0.0", "--port", "8000"]