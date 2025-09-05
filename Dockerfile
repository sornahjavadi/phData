# Use a lightweight Python image
FROM python:3.9-slim

# Work directory inside the container
WORKDIR /app

# Install runtime deps
RUN pip install --no-cache-dir pandas==2.1.1 scikit-learn==1.3.1 fastapi uvicorn

# Copy only what's needed to train & serve
# 1) Data (for training + ZIP-code enrichment)
COPY data ./data
# 2) Training script to produce model artifacts inside the image
COPY create_model.py ./create_model.py

# Train and produce: model/model.pkl + model/model_features.json
RUN python create_model.py

# 3) API & test client
COPY app.py ./app.py
COPY test_api.py ./test_api.py

# Expose FastAPI
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
