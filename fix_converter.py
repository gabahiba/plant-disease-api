import tensorflow as tf

# 1. تحميل النموذج الأصلي
print("جاري تحميل النموذج...")
try:
    model = tf.keras.models.load_model('plant_disease_model_fast.keras')
    print("تم تحميل النموذج بنجاح!")
except Exception as e:
    print(f"خطأ في تحميل النموذج: {e}")
    exit()

# 2. حفظ النموذج مجددًا بصيغة Keras (.keras)
print("جاري حفظ النموذج بصيغة .keras جديدة...")
try:
    tf.keras.models.save_model(model, 'plant_disease_model_fixed.keras', save_format='keras')
    print("تم حفظ النموذج الجديد: plant_disease_model_fixed.keras")
except Exception as e:
    print(f"خطأ في حفظ النموذج: {e}")
    exit()

# 3. (اختياري) طباعة معلومات النموذج للتأكد
print("\nمعلومات النموذج الجديد:")
print(model.summary())

# 4. تأكيد أن النموذج الجديد يعمل
try:
    test_input = tf.random.uniform((1, 128, 128, 3))
    test_output = model.predict(test_input)
    print(f"\n✅ النموذج الجديد يعمل بنجاح! شكل المخرجات: {test_output.shape}")
except Exception as e:
    print(f"\n❌ خطأ في التحقق من النموذج: {e}")