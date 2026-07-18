FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api/ ./api/
COPY models/ ./models/

# Expose port for FastAPI
EXPOSE 8000

# Set environment variable to ensure imports work correctly
ENV PYTHONPATH=/app

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
