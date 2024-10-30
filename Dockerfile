FROM python:3.9-slim
#FROM tensorflow/tensorflow:2.3.0
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
#ENV FLASK_ENV=production

CMD [ "python", "servicenow.py"]
