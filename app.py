from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS
import os
import json

# إنشاء التطبيق
app = Flask(__name__)
CORS(app)

# تحميل الموديل
print("Loading AI model...")
model = load_model("plant_disease_model_fast.keras")
print("Model loaded successfully!")

# تحميل قاعدة بيانات العلاج
with open('disease_treatment.json', 'r', encoding='utf-8') as f:
    treatment_db = json.load(f)

# جلب أسماء الأمراض تلقائياً من مجلد التدريب
train_dir = 'split_dataset/train'

classes = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])

print("Classes loaded:")
print(classes)

# تجهيز الصورة
def preprocess_image(img):

    # نفس حجم التدريب
    img = img.resize((128, 128))

    # تحويل إلى array
    img = np.array(img)

    # normalization
    img = img / 255.0

    # إضافة batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# الصفحة الرئيسية
@app.route("/")
def home():
    return "Plant Disease API Running"

# صفحة تجربة النظام
@app.route("/test")
def test_page():

    return render_template_string("""

    <!DOCTYPE html>
    <html>

    <head>

        <title>Plant Disease AI</title>

        <style>

            body{
                font-family: Arial;
                background:#0f172a;
                color:white;
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
                margin:0;
            }

            .box{
                background:#1e293b;
                padding:40px;
                border-radius:20px;
                text-align:center;
                width:420px;
                box-shadow:0 0 20px rgba(0,0,0,0.3);
            }

            h1{
                margin-bottom:25px;
            }

            input{
                margin-top:15px;
            }

            button{
                background:#22c55e;
                border:none;
                padding:12px 20px;
                color:white;
                border-radius:10px;
                cursor:pointer;
                margin-top:20px;
                font-size:16px;
            }

            button:hover{
                opacity:0.9;
            }

        </style>

    </head>

    <body>

        <div class="box">

            <h1>🌱 Plant Disease Detection</h1>

            <form action="/predict" method="POST" enctype="multipart/form-data">

                <input type="file" name="file" required>

                <br>

                <button type="submit">
                    Analyze Plant
                </button>

            </form>

        </div>

    </body>

    </html>

    """)

# التنبؤ بالمرض
@app.route("/predict", methods=["POST"])
def predict():

    # التحقق من وجود صورة
    if "file" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    try:

        # قراءة الصورة
        file = request.files["file"]

        img = Image.open(file).convert("RGB")

        # تجهيز الصورة
        processed_image = preprocess_image(img)

        # prediction
        prediction = model.predict(processed_image, verbose=0)

        # استخراج أعلى احتمال
        predicted_index = np.argmax(prediction)

        predicted_class = classes[predicted_index]

        # رفض الصور غير النباتية
        if predicted_class == "non_plant":

            return jsonify({

                "status": "rejected",

                "message": "This image is not a plant leaf."

            })

        # التحقق إذا النبات سليم
        is_healthy = "healthy" in predicted_class.lower()

        # جلب معلومات العلاج
        if predicted_class in treatment_db:

            info = treatment_db[predicted_class]

            disease_name = info.get(
                "name_ar",
                predicted_class
            )

            treatment = info.get(
                "treatment",
                "Treatment not available"
            )

        else:

            disease_name = predicted_class

            treatment = "Treatment information not available"

        # النتيجة النهائية
        return jsonify({

            "status": "success",

            "healthy": is_healthy,

            "disease": disease_name,

            "treatment": treatment

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)