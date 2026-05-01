"""
Feedback generation module - Gemini API + Fallback suggestions
Memberikan motivasi/suggestion berdasarkan emosi yang terdeteksi.
"""

import os
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

# Initialize Gemini client (dari env var GOOGLE_API_KEY)
def init_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set - will use fallback suggestions only")
        return None
    try:
        genai.configure(api_key=api_key)
        return True  # Mark as initialized
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
        return None

# Fallback suggestions (jika Gemini API fail atau tidak diset)
FALLBACK_SUGGESTIONS = {
    "angry": "Coba lebih rileks dan tarik napas dalam. Tunjukkan sisi ramahmu kepada interviewer.",
    "disgust": "Alihkan fokus ke hal positif. Kamu punya kendali penuh atas dirimu.",
    "fear": "Tarik napas dalam, kamu sudah mempersiapkan ini dengan baik. Percaya diri!",
    "happy": "Bagus! Pertahankan senyum dan energi positifmu!",
    "neutral": "Tetap tenang dan fokus, jawabanmu terdengar profesional.",
    "sad": "Ayo lebih semangat! Tunjukkan antusiasmemu sedikit lagi.",
    "surprise": "Jangan biarkan kejutan mengganggu fokusmu. Tetap tenang dan fokus!"
}

class FeedbackGenerator:
    def __init__(self):
        self.gemini_client = init_gemini_client()

    def get_suggestion(self, emotion: str) -> str:
        """
        Generate suggestion berdasarkan emosi.
        Coba Gemini API dulu, fallback ke hardcoded suggestions.
        
        Args:
            emotion (str): Nama emosi (lowercase)
        
        Returns:
            str: Suggestion text
        """
        emotion_lower = emotion.lower()
        
        # Try Gemini API jika available
        if self.gemini_client:
            try:
                prompt = (
                    f"Kandidat terlihat {emotion_lower} dalam interview. "
                    f"Berikan 1 kalimat motivasi singkat dalam bahasa Indonesia yang bijaksana dan supportive. "
                    f"Jangan lebih dari 10 kata."
                )
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                suggestion = response.text.strip()
                if suggestion:
                    return suggestion
            except Exception as e:
                logger.debug(f"Gemini API failed: {e}, using fallback")
        
        # Fallback ke hardcoded suggestion
        return FALLBACK_SUGGESTIONS.get(emotion_lower, "Tetap semangat dan fokus!")

# Global instance
_feedback_gen = None

def get_feedback_generator() -> FeedbackGenerator:
    """Get or create global FeedbackGenerator instance"""
    global _feedback_gen
    if _feedback_gen is None:
        _feedback_gen = FeedbackGenerator()
    return _feedback_gen
