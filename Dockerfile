FROM python:3.11-alpine

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/

CMD ["sh", "-c", "if [ -f /app/.env ]; then set -a; . /app/.env; set +a; fi; exec python /app/henry.py"]