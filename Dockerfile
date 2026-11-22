FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and local databases
COPY . .

# Expose ports: 8000 (CRM), 8001 (Tasks), 8002 (Voice Agent)
EXPOSE 8000 8001 8002

# Run the stack
CMD ["python", "start_stack.py"]
