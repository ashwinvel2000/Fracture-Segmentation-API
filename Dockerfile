# 1. Start from official Python image
FROM python:3.10-slim-bullseye

# 2. Set working directory
WORKDIR /app

# 3. Copy only requirements first (for caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 7. Expose port (optional documentation)
EXPOSE 5000

# 8. Default command to run your app
CMD ["python", "main.py"]