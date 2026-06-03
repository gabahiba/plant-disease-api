import tensorflow as tf

# 1. تحميل النموذج الأصلي
print("جاري تحميل النموذج...")
model = tf.keras.models.load_model('plant_disease_model_fast.keras')
print("تم تحميل النموذج بنجاح!")

# 2. حفظ النموذج بصيغة .keras المتوافقة مع إصدار Keras الحديث
print("جاري حفظ النموذج بصيغة .keras جديدة...")
model.save('plant_disease_model_fixed.keras')
print("تم حفظ النموذج الجديد: plant_disease_model_fixed.keras")