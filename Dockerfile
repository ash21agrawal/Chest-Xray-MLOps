FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app.py .
COPY templates templates
COPY saved_models saved_models
COPY uploads uploads

EXPOSE 5000

CMD ["python", "app.py"]