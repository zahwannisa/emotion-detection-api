import os
import io
import base64
import json
import zipfile
import tempfile
import shutil
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
import re

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

class CompatDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('quantization_config', None)
        return super().from_config(config)

class CompatConv2D(tf.keras.layers.Conv2D):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('quantization_config', None)
        return super().from_config(config)

class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('quantization_config', None)
        return super().from_config(config)

class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, filters=None, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        channels = self.filters or input_shape[-1]
        self.filters = channels
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(max(1, channels // self.ratio), activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
        })
        return config

def sanitize_keras_config(config_dict):
    """Recursively remove quantization_config from model config dict (Keras 3 compatibility)"""
    if isinstance(config_dict, dict):
        config_dict.pop('quantization_config', None)
        for v in config_dict.values():
            if isinstance(v, (dict, list)):
                sanitize_keras_config(v)
    elif isinstance(config_dict, list):
        for item in config_dict:
            if isinstance(item, (dict, list)):
                sanitize_keras_config(item)
    return config_dict

# Load model once at startup
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    custom_objects = {
        'LabelSmoothingLoss': LabelSmoothingLoss,
        'SqueezeExcitation': SqueezeExcitation,
        'Dense': CompatDense,
        'Conv2D': CompatConv2D,
        'BatchNormalization': CompatBatchNormalization,
    }
    
    try:
        # Try direct load first
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        logger.info("Model loaded successfully (direct)")
        return model
    except Exception as e:
        logger.info(f"Direct load failed: {type(e).__name__}. Attempting sanitized load...")
        
        try:
            # Extract .keras (ZIP), sanitize config with aggressive regex, reload
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Extracting model to {temp_dir}")
            with zipfile.ZipFile(MODEL_PATH, 'r') as z:
                z.extractall(temp_dir)
                logger.info(f"Files in temp dir: {os.listdir(temp_dir)}")
            
            config_path = None
            for candidate in ('config.json', 'model.json'):
                candidate_path = os.path.join(temp_dir, candidate)
                if os.path.exists(candidate_path):
                    config_path = candidate_path
                    logger.info(f"Found {candidate}, sanitizing...")
                    break

            if config_path is not None:
                # Read raw string and aggressively strip ALL quantization_config occurrences
                with open(config_path, 'r') as f:
                    config_str = f.read()
                
                # Remove ALL variations of quantization_config
                config_str = re.sub(r',\s*"quantization_config"\s*:\s*(?:null|None)', '', config_str)
                config_str = re.sub(r'"quantization_config"\s*:\s*(?:null|None)\s*,\s*', '', config_str)
                config_str = re.sub(r'"quantization_config"\s*:\s*null\s*}', '}', config_str)
                config_str = re.sub(r'"quantization_config"\s*:\s*None\s*}', '}', config_str)
                
                with open(config_path, 'w') as f:
                    f.write(config_str)
                logger.info("Config sanitized (removed all quantization_config)")
            else:
                logger.warning(f"No config.json/model.json found in {temp_dir}")
            
            weights_path = os.path.join(temp_dir, 'model.weights.h5')
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

            logger.info("Building sanitized model from config.json...")
            with open(config_path, 'r') as f:
                sanitized_config_str = f.read()

            model = tf.keras.models.model_from_json(sanitized_config_str, custom_objects=custom_objects)
            logger.info("Loading weights into sanitized model...")
            model.load_weights(weights_path)
            shutil.rmtree(temp_dir)
            logger.info("Model loaded successfully (sanitized)")
            return model
        except Exception as e2:
            logger.error(f"Sanitized load failed: {type(e2).__name__}: {e2}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

# Load model at startup
try:
    model = load_model()
except Exception as e:
    model = None
    logger.warning(f"Model failed to load at startup: {e}")

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
