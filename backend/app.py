import os
import feedparser
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
import io
import base64
import re
import random
import pytesseract
import numpy as np
import cv2
import mediapipe as mp
import torch # PyTorch is a dependency for transformers

# --- APP SETUP ---
# The template_folder points to where your index.html is located.
app = Flask(__name__, template_folder='../frontend/templates')
CORS(app) # Enable Cross-Origin Resource Sharing

# --- In-memory ML Model Loading ---
# This happens once when the server starts.
print("Loading Blip model for Image-to-Speech...")
# Use a smaller, more memory-efficient model if possible for deployment
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Blip model loaded successfully.")

# --- Translation Model Caching ---
translation_models = {}
print("Translation models will be loaded on demand.")

# --- MediaPipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)


# --- In-memory placeholder for database interactions ---
db = {
    "userPreferences": {},
    "usageLogs": [],
}

# --- HELPER FUNCTIONS ---

def get_translation_model(target_lang):
    """Loads and caches a translation model for a specific target language."""
    lang_map = {
        'kn': 'Helsinki-NLP/opus-mt-en-kn', # Kannada
        'hi': 'Helsinki-NLP/opus-mt-en-hi', # Hindi
        'ta': 'Helsinki-NLP/opus-mt-en-ta', # Tamil
        'te': 'Helsinki-NLP/opus-mt-en-te', # Telugu
    }
    model_name = lang_map.get(target_lang)
    if not model_name:
        return None, None
        
    if model_name not in translation_models:
        print(f"Loading translation model for {target_lang}...")
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            translation_models[model_name] = (tokenizer, model)
            print(f"Model for {target_lang} loaded successfully.")
        except Exception as e:
            print(f"Failed to load translation model {model_name}: {e}")
            return None, None
    return translation_models[model_name]

def clean_base64_image(image_data_url):
    """Removes the 'data:image/jpeg;base64,' prefix from the image data URL."""
    return re.sub('^data:image/.+;base64,', '', image_data_url)

def recognize_simple_sign(landmarks):
    """Recognizes simple signs for 'Yes', 'No', and 'Hello'."""
    try:
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

        if (index_tip.y < index_pip.y and middle_tip.y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y < landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y < landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "Hello"
        if thumb_tip.y < index_pip.y and index_tip.y > index_pip.y and pinky_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
            return "Yes"
        if index_tip.y > index_pip.y and pinky_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y and thumb_tip.x > index_tip.x:
            return "No"
    except Exception as e:
        print(f"Error in sign logic: {e}")
    return None


# --- FRONTEND ROUTE ---

@app.route('/')
def index():
    """This route serves your main HTML file."""
    return render_template('index.html')

# --- API ROUTES ---

@app.route('/api/user-preferences/<userId>', methods=['GET', 'POST'])
def user_preferences(userId):
    if request.method == 'POST':
        db["userPreferences"][userId] = request.get_json()
        return jsonify({"message": 'Preferences saved.'}), 200
    else:
        prefs = db["userPreferences"].get(userId)
        return jsonify(prefs) if prefs else (jsonify({"message": "Preferences not found."}), 404)

@app.route('/api/log-usage', methods=['POST'])
def log_usage():
    log_entry = request.get_json()
    log_entry['timestamp'] = __import__('datetime').datetime.now().isoformat()
    db["usageLogs"].append(log_entry)
    print('Usage logged:', log_entry)
    return jsonify({"message": 'Usage logged.'}), 200

@app.route('/api/sos', methods=['POST'])
def sos():
    print('SOS Alert Received:', request.get_json())
    return jsonify({"message": 'SOS alert processed.'}), 200

@app.route('/api/news', methods=['GET'])
def get_news():
    feed_url = 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms'
    try:
        feed = feedparser.parse(feed_url)
        articles = [{"title": entry.title, "source": {"name": getattr(entry, 'author', 'The Times of India')}} for entry in feed.entries[:5]]
        return jsonify({"status": "ok", "articles": articles})
    except Exception as e:
        print(f'Error fetching RSS feed: {e}')
        return jsonify({"status": "error", "message": 'Could not fetch news feed.'}), 500
        
@app.route('/api/predictive-shortcut/<userId>', methods=['GET'])
def predictive_shortcut(userId):
    profile = request.args.get('profileType')
    choices = {
        'visually-impaired': ['Image-to-Speech', 'Voice Assistant', 'Read Document'],
        'elderly': ['Reminders', 'Voice News'],
    }
    default_choice = ['Speech-to-Text', 'Text Translator', 'Text-to-Icon']
    prediction = random.choice(choices.get(profile, default_choice))
    return jsonify({'predicted_feature': prediction})

# --- AI ENDPOINTS ---

@app.route('/api/image-to-speech', methods=['POST'])
def api_image_to_speech():
    try:
        image_data = clean_base64_image(request.json['imageData'])
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/read-document', methods=['POST'])
def api_read_document():
    try:
        image_data = clean_base64_image(request.json['imageData'])
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        extracted_text = pytesseract.image_to_string(image) or "No readable text was found in the image."
        return jsonify({'text': extracted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate-text', methods=['POST'])
def api_translate_text():
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('target_lang', 'en').split('-')[0]
        if target_lang == 'en':
            return jsonify({'translated_text': text})
        
        tokenizer, model = get_translation_model(target_lang)
        if not model:
            return jsonify({'error': f'Translation model for "{target_lang}" is not available.'}), 400

        tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated_tokens = model.generate(**tokenized, max_length=512)
        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return jsonify({'translated_text': translated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-icon', methods=['POST'])
def api_text_to_icon():
    # This logic remains the same as it's just a dictionary lookup
    icon_map = { 'help': '🆘', 'love': '❤️', 'thank you': '🙏', 'yes': '✅', 'no': '❌', 'idea': '💡', 'happy': '😊', 'sad': '😢', 'home': '🏠', 'house': '🏠', 'school': '🏫', 'hospital': '🏥', 'clinic': '🏥', 'pharmacy': '💊', 'shop': '🛒', 'store': '🛒', 'market': '🛒', 'restroom': '🚽', 'toilet': '🚽', 'bank': '🏦', 'post office': '🏤', 'family': '👨‍👩‍👧', 'doctor': '🧑‍⚕️', 'nurse': '🧑‍⚕️', 'teacher': '🧑‍🏫', 'call': '📞', 'phone': '📞', 'talk': '🗣️', 'eat': '🍔', 'food': '🍔', 'drink': '💧', 'water': '💧', 'read': '📖', 'write': '✍️', 'sleep': '😴', 'money': '💰', 'car': '🚗', 'bus': '🚌', 'medicine': '💊', 'pill': '💊', 'book': '📖', 'time': '⏰', 'clock': '⏰', 'today': '📅', 'day': '☀️', 'night': '🌙', }
    text = request.json.get('text', '').lower()
    words = re.sub(r'[^\w\s]', '', text).split()
    icons = []
    found_words = []
    for word in words:
        singular = word.rstrip('s')
        if word in icon_map and word not in found_words:
            icons.append(icon_map[word])
            found_words.append(word)
        elif singular in icon_map and singular not in found_words:
            icons.append(icon_map[singular])
            found_words.append(singular)
    return jsonify({'icons': ' '.join(icons), 'found_words': found_words})

@app.route('/api/recognize-command', methods=['POST'])
def api_recognize_command():
    # This logic remains the same
    intents = { 'find_hospital': ['hospital', 'clinic', 'doctor', 'emergency room', 'medical'], 'call_family': ['call my family', 'phone home', 'contact family'], 'read_news': ['read the news', 'what are the headlines', 'news update'], 'open_translator': ['translate', 'translator'], 'go_home': ['go home', 'back to main screen', 'dashboard'] }
    command = request.json.get('command', '').lower()
    for intent, keywords in intents.items():
        if any(keyword in command for keyword in keywords):
            return jsonify({'intent': intent})
    return jsonify({'intent': 'none'})

@app.route('/api/sign-to-speech', methods=['POST'])
def api_sign_to_speech():
    try:
        base64_image = clean_base64_image(request.json['imageData'])
        img_np = np.frombuffer(base64.b64decode(base64_image), np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        word = "No sign detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                recognized = recognize_simple_sign(hand_landmarks)
                if recognized:
                    word = recognized
                    break
        return jsonify({'word': word})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MAIN RUN ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)

