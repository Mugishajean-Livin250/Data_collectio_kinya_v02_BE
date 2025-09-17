# Use Python 3.11 (compatible with numpy, pydantic, etc.)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for numpy, psycopg2, audio processing, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Expose the port Render will use
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]