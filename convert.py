import tensorflow as tf

model = tf.keras.models.load_model("face_recognition_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # smaller, faster
tflite_model = converter.convert()

with open("face_recognition_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model as face_recognition_model.tflite")
