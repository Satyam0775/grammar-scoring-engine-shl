# 🎙️ Grammar Scoring Engine — SHL Research Intern Task

This project presents a **Grammar Scoring Engine** that predicts the grammatical quality of spoken English using **audio recordings**.  
It uses **Automatic Speech Recognition (ASR)** for transcription and **linguistic feature extraction** (POS-based) to score grammar quality on a **0–5 scale**.

---

## 🚀 Project Overview

| **Component** | **Description** |
|----------------|-----------------|
| **Goal** | Predict grammar score of spoken English (0–5 scale) |
| **Dataset Source** | SHL Grammar Scoring Task (custom dataset) |
| **Core Frameworks** | Whisper, Scikit-Learn, NLTK, Pandas, NumPy |
| **Model Used** | RandomForestRegressor |
| **Final Output** | Grammar score prediction per audio sample |

---

## 🧠 Methodology

1. **Audio Transcription (ASR)**  
   Speech input is converted to text using the **OpenAI Whisper** model.

2. **Feature Extraction**  
   Linguistic features are extracted from transcribed text:
   - Total words  
   - Number of nouns, verbs, adjectives  
   - Average word length  

3. **Model Training**  
   Features are used to train a **Random Forest Regressor**, optimized for grammar score prediction.

4. **Evaluation**  
   Performance evaluated using **Mean Squared Error (MSE)** and **R² Score**.

5. **Deployment**  
   - Google Cloud: Training, feature extraction, and evaluation  
   - Hugging Face Spaces: Interactive Gradio-based web demo

---

## 🧩 Key Features

- Real-time **audio upload** and **grammar scoring**
- **Whisper-based** ASR transcription
- **POS-tagging** linguistic feature extraction (via NLTK)
- **Random Forest regression** for scoring
- Interactive **Gradio UI** hosted on Hugging Face

---

## 📊 Model Performance

| **Metric** | **Score** |
|-------------|-----------|
| Mean Squared Error (MSE) | 0.34 |
| R² Score | 0.72 |

---

## 🧾 Important Links

| **Resource** | **Access Link** |
|---------------|----------------|
| 🧠 **Hugging Face App (Live Demo)** | [https://huggingface.co/spaces/Satyam0077/voice-grammar-scoring-engine](https://huggingface.co/spaces/Satyam0077/voice-grammar-scoring-engine) |
| 💾 **Google Drive Dataset & Training Notebook** | [https://drive.google.com/drive/folders/1L3_Z8_G0FMUwDTd_-Jg__di8milr_hI9?usp=sharing](https://drive.google.com/drive/folders/1L3_Z8_G0FMUwDTd_-Jg__di8milr_hI9?usp=sharing) |
| 🧰 **GitHub Repository** | [https://github.com/Satyam0775/grammar-scoring-engine-shl](https://github.com/Satyam0775/grammar-scoring-engine-shl) |

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Whisper (OpenAI)**
- **NLTK**
- **Scikit-Learn**
- **Gradio**
- **Joblib**
- **Google Cloud Platform (GCP)**

---

## 📦 Outputs Generated

- ✅ `train_progress.csv` — Training set linguistic features  
- ✅ `test_progress.csv` — Test set linguistic features  
- ✅ `grammar_rf_model.pkl` — Trained Random Forest model  
- ✅ `submission.csv` — Final predicted grammar scores  

---

## 🧰 Setup & Usage (Local or Google Colab)

1. Clone the repository:
   ```bash
   git clone https://github.com/Satyam0775/grammar-scoring-engine-shl.git
   cd grammar-scoring-engine-shl

Install dependencies:
pip install -r requirements.txt
Run the notebook:
notebooks/Grammar_Scoring_Engine.ipynb
Generate submission file:

# Inside notebook
model.predict(test_features)
View results in:
dataset/submission.csv

🧪 Deployment Details

Model Training & Inference: Performed on Google Cloud (T4 GPU)
App Hosting: Hugging Face Spaces with Gradio Interface
Scoring: Based on extracted linguistic features (POS-based)
Prediction: RandomForestRegressor trained using SHL dataset

🧑‍💻 Author
Satyam Kumar
Data Science & AI Enthusiast