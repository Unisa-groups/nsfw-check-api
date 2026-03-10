FROM python:3.12.13-bookworm
LABEL authors="ed"
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY download_model.py ./
RUN python3 download_model.py
COPY . .
ENV MODEL_PATH=/usr/src/app/model
CMD [ "python", "./main.py" ]
