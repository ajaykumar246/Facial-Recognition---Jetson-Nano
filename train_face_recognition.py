import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("✅ TensorFlow version:", tf.__version__)

# ✅ GPU detection for TF 1.15
from tensorflow.python.client import device_lib
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
print("✅ GPU Available:", gpus if gpus else "❌ No GPU found")

# Base directory
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "train")  # Only one folder

# Parameters (optimized for Jetson Nano 2GB)
img_height, img_width = 96, 96
batch_size = 8
epochs = 5
learning_rate = 0.0001

# Auto-split the single dataset into train/validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,   # 20% validation
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation'
)

# Build lightweight model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1
)

# Save model
model.save("face_recognition_model.h5")
print("✅ Training complete — model saved as face_recognition_model.h5")
