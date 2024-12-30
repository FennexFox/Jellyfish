import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import numpy as np
import os
from io import BytesIO

def load_model():
    model_path = os.path.expanduser('~/keras/Jellyfish/vgg16.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

async def prediction_model(model, img_data):
    try:
        img = Image.open(BytesIO(img_data))
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        raise Exception(f"이미지 처리 오류: {e}")
    
    target_size = (224, 224)  # VGG16 입력 크기
    
    img_to_process = img.resize(target_size)
    numpy_img = np.array(img_to_process)
    img_batch = np.expand_dims(numpy_img, axis=0)
    pre_processed = preprocess_input(img_batch)
    
    try:
        predictions = model.predict(pre_processed)
        top_predictions = decode_predictions(predictions, top=3)[0]
        
        result = [
            {
                "predicted_label": str(pred[1]),
                "prediction_score": float(pred[2])
            }
            for pred in top_predictions
        ]
        
        return {"result": result}
    except Exception as e:
        raise Exception(f"예측 처리 오류: {e}")