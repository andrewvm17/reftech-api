# Dockerfile

FROM python:3.10-slim

# 1) Install system packages for OpenCV (e.g. libGL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # (Optional) Additional packages you may need for OpenCV
    libsm6 \
    libxext6 \
    libxrender1 \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# 2) Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Set work directory
WORKDIR /app

# 4) Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5) Copy project files
COPY . .

# 6) Expose port 5000
EXPOSE 5000

# 7) Use gunicorn to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
