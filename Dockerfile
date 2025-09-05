FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas==2.1.1 scikit-learn==1.3.1 fastapi uvicorn

COPY data ./data
COPY create_model.py ./create_model.py

RUN python create_model.py

COPY app.py ./app.py
COPY test_api.py ./test_api.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
