FROM python:3.10-slim

WORKDIR /app

# Copy requirements from repo root if it exists
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Dash app code
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]