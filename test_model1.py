import tensorflow as tf
model = tf.keras.models.load_model('plant_disease_model_fixed.keras')
print("عدد فئات النموذج:", model.output_shape[-1])