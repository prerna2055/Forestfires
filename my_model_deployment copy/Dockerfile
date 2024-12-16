# Use the official Python image as the base image
FROM python:3.12.3

# Set the working directory in the container
WORKDIR /my_model_deployment2

# Copy requirements.txt into the container
WORKDIR /app

COPY requirements_.txt .

RUN pip install --no-cache-dir -r requirements_.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]

