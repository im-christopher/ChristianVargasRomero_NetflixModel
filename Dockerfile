FROM python:3.10-slim
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./app.py .
COPY ./model ./model
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
