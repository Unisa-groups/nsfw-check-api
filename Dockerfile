FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py ./
RUN python3 download_model.py

COPY . .

ENV MODEL_PATH=/usr/src/app/model
EXPOSE 8123

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
