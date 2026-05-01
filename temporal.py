"""
Temporal smoothing module - Aggregate emotion predictions over time windows
Mengurangi noise dan jittering dalam deteksi emosi.
"""

import numpy as np
from collections import deque
from typing import List, Tuple
import time

class EmotionTemporalAggregator:
    """
    Aggregate emotion predictions over a time window (default 3 detik).
    Smooth predictions dan reduce false positives.
    
    Attributes:
        window_size (int): Berapa banyak frames dalam satu window
        fps (int): Frame per second dari input (default 30)
        time_window_sec (float): Time window dalam detik (akan dihitung dari fps)
        prediction_buffer (deque): Buffer untuk store predictions dalam window
    """
    
    def __init__(self, time_window_sec: float = 3.0, fps: int = 30):
        """
        Initialize aggregator.
        
        Args:
            time_window_sec (float): Desired time window dalam detik (default 3.0)
            fps (int): Frame per second (default 30)
        """
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.window_size = max(1, int(time_window_sec * fps))  # Berapa frame dalam window
        self.prediction_buffer = deque(maxlen=self.window_size)
        self.emotion_scores_buffer = deque(maxlen=self.window_size)  # Scores dari model
        
    def add_prediction(self, emotion_label: str, emotion_scores: np.ndarray):
        """
        Add satu prediction ke buffer.
        
        Args:
            emotion_label (str): Predicted emotion label (e.g., 'happy')
            emotion_scores (np.ndarray): Array of scores dari model (e.g., [0.1, 0.05, 0.02, 0.75, 0.05, 0.01, 0.01])
        """
        self.prediction_buffer.append(emotion_label)
        self.emotion_scores_buffer.append(emotion_scores)
    
    def is_window_full(self) -> bool:
        """Check apakah buffer sudah full (siap untuk aggregate)"""
        return len(self.prediction_buffer) >= self.window_size
    
    def get_aggregated_emotion(self) -> Tuple[str, float, np.ndarray]:
        """
        Get aggregated emotion berdasarkan predictions dalam buffer.
        
        Strategi:
        1. Average scores dari semua predictions dalam window
        2. Ambil emotion dengan highest average score
        3. Return emotion label + confidence + scores
        
        Returns:
            Tuple[str, float, np.ndarray]: (emotion_label, confidence, averaged_scores)
        
        Raises:
            ValueError: Jika buffer kosong
        """
        if len(self.emotion_scores_buffer) == 0:
            raise ValueError("Prediction buffer kosong, tidak bisa aggregate")
        
        # Average scores
        scores_array = np.array(list(self.emotion_scores_buffer))  # Shape: (N, 7) untuk 7 emosi
        avg_scores = np.mean(scores_array, axis=0)  # Shape: (7,)
        
        # Highest confidence
        emotion_idx = np.argmax(avg_scores)
        confidence = float(avg_scores[emotion_idx])
        
        # Get emotion label dari buffer (most frequent dalam window)
        # atau bisa dari emotion_idx
        emotion_labels = list(self.prediction_buffer)
        # Alternative: ambil label dari emotion_idx (jika punya mapping)
        
        return emotion_labels[emotion_idx], confidence, avg_scores
    
    def get_smoothed_emotion(self, emotion_mapping: List[str]) -> Tuple[str, float]:
        """
        Get smoothed emotion dengan mapping index ke label.
        
        Args:
            emotion_mapping (List[str]): List emotion labels (e.g., ['angry', 'disgust', 'fear', 'happy', ...])
        
        Returns:
            Tuple[str, float]: (emotion_label, confidence)
        """
        if len(self.emotion_scores_buffer) == 0:
            return "neutral", 0.0
        
        scores_array = np.array(list(self.emotion_scores_buffer))
        avg_scores = np.mean(scores_array, axis=0)
        
        emotion_idx = np.argmax(avg_scores)
        confidence = float(avg_scores[emotion_idx])
        emotion_label = emotion_mapping[emotion_idx] if emotion_idx < len(emotion_mapping) else "unknown"
        
        return emotion_label, confidence
    
    def reset(self):
        """Reset buffers"""
        self.prediction_buffer.clear()
        self.emotion_scores_buffer.clear()
    
    def get_buffer_stats(self) -> dict:
        """Get statistics tentang current buffer"""
        return {
            "buffer_size": len(self.prediction_buffer),
            "window_size": self.window_size,
            "is_full": self.is_window_full(),
            "time_window_sec": self.time_window_sec,
            "fps": self.fps
        }


class EmotionWindowManager:
    """
    Manage multiple windows untuk continuous emotion detection.
    Sliding window approach: agregasi setiap N frame, emit result.
    
    Contoh:
    - Window size 3 detik @ 30 fps = 90 frames
    - Setiap 30 frame (1 detik), emit 1 aggregated result (sliding window)
    - Atau emit per window_size frames (non-overlapping)
    """
    
    def __init__(self, time_window_sec: float = 3.0, fps: int = 30, slide_ratio: float = 1.0):
        """
        Initialize window manager.
        
        Args:
            time_window_sec (float): Desired time window dalam detik
            fps (int): Frame per second
            slide_ratio (float): Ratio slide (1.0 = non-overlapping, 0.5 = 50% overlap)
        """
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.window_size = max(1, int(time_window_sec * fps))
        self.slide_size = max(1, int(self.window_size * slide_ratio))
        self.frame_count = 0
        self.prediction_buffer = deque(maxlen=self.window_size)
        self.emotion_scores_buffer = deque(maxlen=self.window_size)
    
    def add_frame_prediction(self, emotion_label: str, emotion_scores: np.ndarray) -> Tuple[bool, dict]:
        """
        Add satu frame prediction.
        
        Returns:
            Tuple[bool, dict]: (should_emit, result_dict)
                - should_emit: True jika sudah time untuk emit aggregated result
                - result_dict: dict dengan emotion, confidence, dll (jika should_emit=True)
        """
        self.prediction_buffer.append(emotion_label)
        self.emotion_scores_buffer.append(emotion_scores)
        self.frame_count += 1
        
        should_emit = (self.frame_count % self.slide_size == 0) and len(self.prediction_buffer) >= self.window_size
        
        if should_emit:
            result = self._get_window_result()
            return True, result
        
        return False, {}
    
    def _get_window_result(self) -> dict:
        """Calculate aggregated result untuk current window"""
        if len(self.emotion_scores_buffer) == 0:
            return {"emotion": "neutral", "confidence": 0.0, "frame_count": 0}
        
        scores_array = np.array(list(self.emotion_scores_buffer))
        avg_scores = np.mean(scores_array, axis=0)
        
        emotion_idx = np.argmax(avg_scores)
        confidence = float(avg_scores[emotion_idx])
        
        return {
            "emotion_idx": emotion_idx,
            "avg_scores": avg_scores,
            "confidence": confidence,
            "frame_count": len(self.prediction_buffer),
            "window_sec": self.time_window_sec
        }
