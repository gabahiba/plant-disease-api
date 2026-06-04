from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS
import os
import json

# تفعيل التنفيذ المتعجل (قد يساعد في الأداء لكن ليس ضرورياً)
tf.config.run_functions_eagerly(True)

app = Flask(__name__)
CORS(app)

# ===========================
# 1. تحميل نموذج TFLite
# ===========================
print("جاري تحميل نموذج TFLite...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# الحصول على تفاصيل الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ تم تحميل نموذج TFLite بنجاح.")

# ===========================
# 2. تحميل قائمة الفئات (من class_order.json)
# ===========================
with open('class_order.json', 'r', encoding='utf-8') as f:
    classes = json.load(f)
print(f"✅ تم تحميل {len(classes)} فئة بالترتيب الصحيح.")

# ===========================
# 3. تحميل قاعدة بيانات العلاج
# ===========================
with open('disease_treatment.json', 'r', encoding='utf-8') as f:
    treatment_db = json.load(f)
print("تم تحميل قاعدة بيانات العلاج.")

# ===========================
# 4. دالة تجهيز الصورة
# ===========================
def preprocess_image(img):
    # تغيير الحجم كما في التدريب (128×128)
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    # إضافة بُعد الدفعة (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===========================
# 5. دوال المسارات (Endpoints)
# ===========================
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
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not classes:
        return jsonify({"error": "Model not configured properly (no classes loaded)"}), 500

    try:
        # قراءة الصورة ومعالجتها
        img = Image.open(file).convert("RGB")
        processed_img = preprocess_image(img)

        # التنبؤ باستخدام TFLite
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        if predicted_index >= len(classes):
            return jsonify({"error": f"Prediction index {predicted_index} out of range"}), 500

        predicted_class = classes[predicted_index]

        # رفض الصور غير النباتية إذا كانت الفئة موجودة
        if predicted_class == "non_plant":
            return jsonify({
                "status": "rejected",
                "message": "This image is not a plant leaf."
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
            "treatment": treatment
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)