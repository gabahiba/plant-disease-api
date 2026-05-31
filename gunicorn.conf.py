import os

# ربط التطبيق بالمنفذ المحدد من Render أو 10000 بشكل افتراضي
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = 1  # عدد العمال، يمكنك زيادته إذا أردت