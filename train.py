from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
from flask_cors import CORS
from tensorflow.keras.models import save_model
save_model(model, 'plant_disease_model_fixed.keras', save_format='keras')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# تحميل النموذج
print("Loading model...")
model = tf.keras.models.load_model('plant_disease_model_fast.keras')
print("Model loaded successfully!")

# تحميل قاعدة بيانات العلاج
with open('disease_treatment.json', 'r', encoding='utf-8') as f:
    treatment_db = json.load(f)

# أسماء الفئات
train_dir = 'split_dataset/train'

class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])

print(f"Number of classes: {len(class_names)}")

# معالجة الصورة
def preprocess_image(img):

    img = img.resize((128, 128))

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# الصفحة الرئيسية
@app.route("/")
def home():
    return "Plant Disease API Running"

# API prediction
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["file"]

    try:
        # فتح الصورة
        img = Image.open(file).convert("RGB")

        # تجهيز الصورة
        processed_img = preprocess_image(img)

        # prediction
        pred = model.predict(processed_img, verbose=0)[0]

        class_idx = np.argmax(pred)

        confidence = float(np.max(pred))

        predicted_class = class_names[class_idx]

        # إذا الصورة ليست نبتة
        if predicted_class == 'non_plant':

            return jsonify({
                "status": "rejected",
                "message": "The uploaded image is not a plant leaf."
            })

        # التحقق إذا النبات سليم
        is_healthy = "healthy" in predicted_class.lower()

        # معلومات العلاج
        if predicted_class in treatment_db:

            info = treatment_db[predicted_class]

            treatment = info.get(
                'treatment',
                'Treatment not available'
            )

            disease_name = info.get(
                'name_ar',
                predicted_class
            )

        else:

            treatment = "Treatment information not available"

            disease_name = predicted_class

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

# تشغيل السيرفر
if __name__ == "__main__":
    app.run(debug=True)