FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all necessary files to the container
COPY . /app

# Ensure the model weights are copied correctly
COPY model_weights.pth /app/model/model_weights.pth

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]