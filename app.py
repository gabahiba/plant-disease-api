from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import InputLayer

tf.config.run_functions_eagerly(True)

app = Flask(__name__)
CORS(app)

# 1. تحميل الموديل بصيغة SavedModel
print("جاري تحميل النموذج من مجلد SavedModel...")
try:
    model = tf.keras.models.load_model('plant_disease_model_fixed.keras', custom_objects={'InputLayer': InputLayer})
    print("✅ تم تحميل النموذج بنجاح باستخدام custom_objects.")
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    model = None

# قائمة الفئات الثابتة (يجب أن تكون بنفس الترتيب الذي تم التدريب عليه)
# استبدال القائمة اليدوية بقراءة الترتيب من ملف JSON
import json
with open('class_order.json', 'r', encoding='utf-8') as f:
    classes = json.load(f)
print(f"✅ تم تحميل {len(classes)} فئة بالترتيب الصحيح.")
print("تم تحميل قائمة الفئات.")

with open('disease_treatment.json', 'r', encoding='utf-8') as f:
    treatment_db = json.load(f)
print("تم تحميل قاعدة بيانات العلاج.")

def preprocess_image(img):
    # نفس حجم التدريب
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# دوال الـ endpoints ('/', '/test', '/predict') بنفس الشكل الذي كانت عليه...
# ... (لنطبق عليهم نفس الكود القديم ولكن مع المتغيرات الجديدة)
@app.route("/")
def home():
    return "Plant Disease API Running"

@app.route("/test")
def test_page():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>Plant Disease AI</title>
    <style>
        body{font-family: Arial; background:#0f172a; color:white; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;}
        .box{background:#1e293b; padding:40px; border-radius:20px; text-align:center; width:420px; box-shadow:0 0 20px rgba(0,0,0,0.3);}
        h1{margin-bottom:25px;}
        input{margin-top:15px;}
        button{background:#22c55e; border:none; padding:12px 20px; color:white; border-radius:10px; cursor:pointer; margin-top:20px; font-size:16px;}
        button:hover{opacity:0.9;}
    </style>
    </head>
    <body>
    <div class="box">
        <h1>🌱 Plant Disease Detection</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <button type="submit">Analyze Plant</button>
        </form>
    </div>
    </body>
    </html>
    """)

@app.route("/predict", methods=["POST"])
def predict():
    import time
    start_time = time.time()
    try:
        # ... الكود الموجود داخل الدالة ...
        pass  # استبدلي هذا بالكود الأصلي للدالة
    except Exception as e:
        import traceback
        print(f"❌ خطأ في التنبؤ: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        end_time = time.time()
        print(f"✅ وقت معالجة الطلب: {end_time - start_time:.2f} ثانية")
        
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # التأكد من أن قائمة الفئات ليست فارغة
    if not classes:
        return jsonify({"error": "Model not configured properly (no classes loaded)"}), 500

    try:
        img = Image.open(file).convert("RGB")
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image, verbose=0)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if predicted_index >= len(classes):
            return jsonify({"error": f"Prediction index {predicted_index} out of range"}), 500

        predicted_class = classes[predicted_index]

        if predicted_class == "non_plant":
            return jsonify({
                "status": "rejected",
                "message": "This image is not a plant leaf.",
                "confidence": confidence
            })

        is_healthy = "healthy" in predicted_class.lower()

        if predicted_class in treatment_db:
            info = treatment_db[predicted_class]
            disease_name = info.get("name_ar", predicted_class)
            treatment = info.get("treatment", "Treatment not available")
        else:
            disease_name = predicted_class
            treatment = "Treatment information not available"

        return jsonify({
            "status": "success",
            "healthy": is_healthy,
            "disease": disease_name,
            "treatment": treatment,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)