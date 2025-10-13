import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
def face_check():
	# Disable all TF warnings / logs for speed
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	
	# ===== TensorFlow 1.x memory optimization =====
	from tensorflow.compat.v1 import ConfigProto, InteractiveSession
	config = ConfigProto()
	config.gpu_options.allow_growth = True   # Use only as much GPU RAM as needed
	config.intra_op_parallelism_threads = 1
	config.inter_op_parallelism_threads = 1
	session = InteractiveSession(config=config)
	
	# ===== Load Model =====
	print("🔹 Loading model...")
	model = load_model("face_recognition_model.h5")
	print("✅ Model loaded successfully")
	
	# ===== Camera Setup (V4L2 backend) =====
	CAMERA_ID = "/dev/video1"
	WIDTH, HEIGHT = 640, 480
	cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
	cap.set(cv2.CAP_PROP_FPS, 30)
	
	# ===== Labels (auto-load from directory) =====
	train_dir = os.path.join(os.getcwd(), "train")
	class_names = sorted(os.listdir(train_dir))
	print("🧠 Classes:", class_names)
	
	# ===== Image Preprocessing =====
	IMG_HEIGHT, IMG_WIDTH = 96, 96  # same as training
	
	def preprocess_frame(frame):
	    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
	    img = img.astype("float32") / 255.0
	    return np.expand_dims(img, axis=0)
	
	# ===== Main Loop =====
	print("\n🎥 Press '1' to predict | 'q' or ESC to quit\n")
	
	while cap.isOpened():
	    ret, frame = cap.read()
	    if not ret:
	        print("❌ Frame capture failed.")
	        break
	
	    cv2.imshow("Jetson Face Recognition", frame)
	    key = cv2.waitKey(10) & 0xFF
	
	    if key == ord('1'):
	        # Start timing for performance check
	        start = time.time()
	        input_img = preprocess_frame(frame)
	        preds = model.predict(input_img)
	        pred_idx = np.argmax(preds)
	        confidence = preds[0][pred_idx]
	        label = class_names[pred_idx]
	        fps = 1 / (time.time() - start)
	
	        # Display result
	        text = f"{label}: {confidence*100:.1f}% ({fps:.1f} FPS)"
	        print("🟩", text)
	        cv2.putText(frame, text, (20, 40),
	                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	        cv2.imshow("Jetson Face Recognition", frame)
	    elif key == 27 or key == ord('q'):
	        print("👋 Exiting...")
	        break
	
	cap.release()
	cv2.destroyAllWindows()
