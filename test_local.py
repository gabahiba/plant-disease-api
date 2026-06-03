import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. تحميل النموذج الجديد
print("جاري تحميل النموذج الجديد محليًا...")
try:
    model = tf.keras.models.load_model('plant_disease_model_fixed.keras')
    print("✅ تم تحميل النموذج الجديد بنجاح محليًا! النموذج جاهز للرفع إلى Render.")
except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل النموذج: {e}")
    exit()

# (اختياري) إذا أردتِ التأكد من قدرته على التنبؤ
# يمكنكِ إلغاء التعليق على الأسطر التالية، لكن استبدلي 'path_to_image.jpg' بمسار صورة حقيقية
"""
img_path = "path_to_image.jpg"  # غيري هذا المسار
if os.path.exists(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    print("تم التنبؤ بنجاح!")
"""