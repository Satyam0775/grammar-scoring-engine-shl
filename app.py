import gradio as gr
import whisper
import joblib
import soundfile as sf
import numpy as np
import pandas as pd
import tempfile
import os
import spacy


# 1️⃣ LOAD MODELS (Whisper + Random Forest)
print("🔹 Loading Whisper model (tiny)...")
asr_model = whisper.load_model("tiny")
print("✅ Whisper model loaded successfully!")

print("🔹 Loading Grammar Random Forest model...")
rf_model = joblib.load("grammar_rf_model.pkl")
print("✅ Grammar model loaded successfully!")

# 2️⃣ LOAD SPACY MODEL (NO NLTK NEEDED)
print("🔹 Loading spaCy model...")
spacy_model = "en_core_web_sm"

try:
    nlp = spacy.load(spacy_model)
except:
    # if not present in Hugging Face container, download it
    os.system(f"python -m spacy download {spacy_model}")
    nlp = spacy.load(spacy_model)

print("✅ spaCy model loaded successfully!")

# 3️⃣ FEATURE EXTRACTION (Using spaCy)

def extract_features_from_audio(audio_path):
    """Transcribe audio using Whisper and extract linguistic features using spaCy."""
    result = asr_model.transcribe(audio_path, fp16=False)
    text = result.get("text", "").strip()

    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]

    total_words = len(tokens)
    num_nouns = len([t for t in doc if t.pos_ == "NOUN"])
    num_verbs = len([t for t in doc if t.pos_ == "VERB"])
    num_adjs = len([t for t in doc if t.pos_ == "ADJ"])
    avg_word_len = np.mean([len(t) for t in tokens]) if total_words > 0 else 0

    features = {
        "total_words": total_words,
        "num_nouns": num_nouns,
        "num_verbs": num_verbs,
        "num_adjs": num_adjs,
        "avg_word_len": avg_word_len,
    }

    print("✅ Extracted features:", features)
    return text, features

# 4️⃣ PREDICTION FUNCTION

def predict_grammar_score(audio):
    """Main Gradio pipeline: Audio → Whisper → spaCy → Grammar Score"""
    try:
        if audio is None:
            return "Please upload an audio file!", "", ""

        sr, data = audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, data, sr)
            audio_path = tmp.name

        # Extract features
        text, features = extract_features_from_audio(audio_path)
        X = pd.DataFrame([features])

        # Predict
        score = rf_model.predict(X)[0]
        feature_summary = "\n".join([f"{k}: {v}" for k, v in features.items()])

        print(f"✅ Prediction successful — Score: {score:.2f}")
        return text, feature_summary, f"{score:.2f} / 5.0"

    except Exception as e:
        print("❌ Exception during prediction:", e)
        return f"❌ Error: {str(e)}", "", ""

# 5️⃣ GRADIO INTERFACE

interface = gr.Interface(
    fn=predict_grammar_score,
    inputs=gr.Audio(type="numpy", label="🎤 Upload or Record English Speech (WAV/MP3)"),
    outputs=[
        gr.Textbox(label="🗣️ Transcription"),
        gr.Textbox(label="📊 Linguistic Features"),
        gr.Textbox(label="🧮 Predicted Grammar Score (0–5 Scale)")
    ],
    title="🎙️ Grammar Scoring Engine — Satyam’s SHL Project",
    description="Upload or record English speech. This app transcribes the audio using Whisper, extracts linguistic features (via spaCy), and predicts a grammar score using a Random Forest model trained on the SHL dataset.",
    theme="default"
)

if __name__ == "__main__":
    print("🚀 Launching app on Hugging Face...")
    interface.launch(server_name="0.0.0.0", server_port=7860)
