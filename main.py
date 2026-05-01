import os
import io
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
from feedback import get_feedback_generator
from temporal import EmotionTemporalAggregator
import logging

from dotenv import load_dotenv

# Load environment variables from .env next to this file (if present)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config via environment
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/emotion_detection_v1.keras')
API_KEY = os.environ.get('API_KEY', 'changeme')
IMG_SIZE = (48, 48)
CLASS_NAMES = ['angry','disgust','fear','happy','neutral','sad','surprise']
MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimal confidence untuk dianggap valid prediction

app = FastAPI(title="Emotion Detection API", version="1.0")

# Reimplement minimal custom objects used by the model so load_model works
class LabelSmoothingLoss(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.1, name='label_smoothing_loss'):
        super().__init__(name=name)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        smooth_positives = 1.0 - self.smoothing
        smooth_negatives = self.smoothing / num_classes
        y_true_smoothed = y_true * smooth_positives + smooth_negatives
        return tf.keras.losses.categorical_crossentropy(y_true_smoothed, y_pred)

class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(max(1, channels // self.ratio), activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * x

# Load model once at startup
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    custom_objects = {
        'LabelSmoothingLoss': LabelSmoothingLoss,
        'SqueezeExcitation': SqueezeExcitation
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    return model

try:
    model = load_model()
except Exception as e:
    model = None
    print("Warning: model failed to load at startup:", e)

# Initialize feedback generator
feedback_gen = get_feedback_generator()

# API key dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API Key')
    return True

class PredictResponse(BaseModel):
    predicted_class: str
    scores: List[float]
    confidence: float
    suggestion: Optional[str] = None  # Motivasi/feedback dari Gemini atau fallback
    meets_confidence_threshold: bool  # Apakah confidence >= MIN_CONFIDENCE_THRESHOLD

@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict', response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Model load error: {e}')

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('L')  # grayscale
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image file')

    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    arr = np.expand_dims(arr, axis=0)   # (1,H,W,1)

    preds = model.predict(arr)
    scores = preds[0].tolist()
    idx = int(np.argmax(scores))
    confidence = float(np.max(scores))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    
    # Check confidence threshold
    meets_threshold = confidence >= MIN_CONFIDENCE_THRESHOLD
    
    # Generate suggestion berdasarkan emosi (dengan atau tanpa Gemini)
    suggestion = None
    if meets_threshold:
        try:
            suggestion = feedback_gen.get_suggestion(label)
        except Exception as e:
            logger.warning(f"Failed to generate suggestion: {e}")
            suggestion = None

    return PredictResponse(
        predicted_class=label,
        scores=scores,
        confidence=confidence,
        suggestion=suggestion,
        meets_confidence_threshold=meets_threshold
    )

@app.post('/reload', dependencies=[Depends(verify_api_key)])
async def reload_model():
    global model
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to reload model: {e}')
    return {'status': 'reloaded'}

@app.get('/')
async def root():
    return {'message': 'Emotion Detection API. Use /predict with X-API-KEY header.'}
