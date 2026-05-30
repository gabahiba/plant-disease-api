FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ بقية ملفات المشروع
COPY . .

# التأكد من أن المنفذ 10000 مكشوف
EXPOSE 10000

# تشغيل التطبيق باستخدام gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
