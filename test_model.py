import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from tkinter import filedialog
from tkinter import Tk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# تحميل النموذج الجديد (الذي تدرب على 40 فئة)
print("جاري تحميل النموذج...")
model = tf.keras.models.load_model('plant_disease_model_fast.keras')
print("تم التحميل.")

# تحميل قاعدة بيانات العلاج
with open('disease_treatment.json', 'r', encoding='utf-8') as f:
    treatment_db = json.load(f)

# أسماء الفئات من مجلد التدريب (يجب أن تحتوي على non_plant)
train_dir = 'split_dataset/train'
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"عدد الفئات: {len(class_names)} (يجب أن تشمل non_plant)")

def predict_image(img_path):
    # حجم الصورة يجب أن يتطابق مع حجم التدريب (128×128)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    predicted_class = class_names[class_idx]
    return predicted_class, confidence

# اختيار الصورة
root = Tk()
root.withdraw()
img_path = filedialog.askopenfilename(title="اختر صورة (يفضل ورقة نبات)", 
                                      filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if img_path:
    print(f"\nالصورة: {img_path}")
    disease, conf = predict_image(img_path)
    
    # إذا كانت النتيجة non_plant، نرفض الصورة
    if disease == 'non_plant':
        print("\n🚫 **النظام يرفض هذه الصورة**")
        print(f"   التصنيف: {disease} (ثقة: {conf:.2%})")
        print("   السبب: الصورة لا تمثل ورقة نبات.")
    else:
        is_healthy = "healthy" in disease.lower()
        print(f"\n🔍 النتيجة: {disease}")
        print(f"📊 الثقة: {conf:.2%}")
        
        if disease in treatment_db:
            info = treatment_db[disease]
            print(f"\n🌿 الاسم: {info.get('name_ar', disease)}")
            print(f"💊 العلاج: {info.get('treatment', 'غير متوفر')}")
        else:
            print("\n⚠️ معلومات العلاج غير متوفرة لهذا المرض في قاعدة البيانات.")
        
        if is_healthy:
            print("\n✅ النبات سليم، لا حاجة للعلاج.")
        else:
            print("\n⚠️ النبات مريض، يرجى اتباع التعليمات أعلاه.")
else:
    print("لم تختر صورة.")