FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some ML packages (e.g. LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements-dev.txt ./

RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy project source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e ".[dev]"

# Default: start JupyterLab on port 8888
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
