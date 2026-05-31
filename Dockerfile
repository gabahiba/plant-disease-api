FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# الأمر المعدل: يربط الخادوم بالعنوان 0.0.0.0 ويستخدم المنفذ المُعطى
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]