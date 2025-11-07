# 🎙️ Grammar Scoring Engine — SHL Research Intern Task

This project presents a **Grammar Scoring Engine** that predicts the grammatical quality of spoken English using **audio recordings**.  
It uses **Automatic Speech Recognition (ASR)** for transcription and **acoustic & linguistic feature extraction** to score grammar quality on a **0–5 scale**.

---

## 🧾 Important Links

| **Resource** | **Access Link** |
|---------------|----------------|
| 🧠 **Hugging Face App (Live Demo)** | [https://huggingface.co/spaces/Satyam0077/voice-grammar-scoring-engine](https://huggingface.co/spaces/Satyam0077/voice-grammar-scoring-engine) |
| 💾 **Google Drive Dataset & Training Notebook** | [https://drive.google.com/drive/folders/1L3_Z8_G0FMUwDTd_-Jg__di8milr_hI9?usp=sharing](https://drive.google.com/drive/folders/1L3_Z8_G0FMUwDTd_-Jg__di8milr_hI9?usp=sharing) |

---

## 🚀 Project Overview

🏆 This project was developed as part of the **SHL Intern Hiring Assessment 2025**, where the goal is to predict grammar scores (0–5 scale) from spoken English audios.

| **Component** | **Description** |
|----------------|-----------------|
| **Goal** | Predict grammar score of spoken English (0–5 scale) |
| **Dataset Source** | SHL Grammar Scoring Task (custom dataset) |
| **Core Frameworks** | Librosa, Scikit-Learn, Pandas, NumPy |
| **Model Used** | RandomForestRegressor (Scikit-Learn) |
| **Evaluation Metrics** | RMSE, Pearson Correlation |
| **Final Output** | `submission_final_kaggle_FIXED_for_kaggle.csv` (197 rows, 2 columns: filename, label) |

---

## 🧠 Methodology

1. **Audio Feature Extraction (Librosa)**  
   Each `.wav` file is processed to extract 5 key features:  
   - MFCC Mean  
   - Spectral Centroid  
   - Spectral Bandwidth  
   - Zero Crossing Rate  
   - Root Mean Square (RMS) Energy  

2. **Feature Normalization**  
   StandardScaler is used to normalize extracted features.

3. **Model Training**  
   A **Random Forest Regressor** is trained on extracted features to predict grammar scores.

4. **Evaluation**  
   Metrics used: **RMSE (Root Mean Squared Error)** and **Pearson Correlation**.  
   These capture both accuracy and score consistency.

5. **Visualization**  
   Scatter plots between actual and predicted grammar scores demonstrate model fit and reliability.

6. **Deployment**  
   - Model trained on Google Colab (T4 GPU).  
   - Deployed via Hugging Face for real-time prediction.

---

## 📊 Model Performance

| **Metric** | **Score** |
|-------------|-----------|
| **Train RMSE** | 0.4613 |
| **Train Pearson Correlation** | 0.9734 |
| **Validation RMSE (local)** | ~0.35 |
| **Validation Pearson Correlation (local)** | ~0.76 |

**Interpretation:**  
The model shows strong correlation between predicted and actual grammar scores with low RMSE, indicating consistent performance.  
Training correlation of ~0.97 confirms model reliability, while validation correlation of ~0.76 shows good generalization to unseen data.

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
