"""
Training del modello di Language Detection per MuseumLangID

Questo script addestra il modello Multinomial Naive Bayes per identificare
la lingua (IT/EN/DE) delle descrizioni museali e lo salva come pickle.
"""

import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def preprocess_text(text):
    """
    Preprocessing specifico per language detection.
    Mantiene apostrofi (distintivi per italiano), rimuove resto punteggiatura.
    """
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)  # Rimuovi numeri
    text = re.sub(r"[^\w\s']", ' ', text)  # Mantieni solo lettere, spazi, apostrofi
    text = re.sub(r'\s+', ' ', text).strip()  # Normalizza spazi
    return text


# Caricamento dataset
print("Caricamento dataset...")
url = "https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv"
df = pd.read_csv(url)
print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")

# Preprocessing
print("\nPreprocessing testi...")
df['Testo_Pulito'] = df['Testo'].apply(preprocess_text)
print(f"Testi preprocessati. Lunghezza media: {df['Testo_Pulito'].str.len().mean():.1f} caratteri")

# Feature extraction con TF-IDF e character 3-grams
print("\nFeature extraction (character 3-grams + TF-IDF)...")
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 3),
    lowercase=False,  # Già fatto nel preprocessing
    min_df=1,
    max_df=1.0
)

X = vectorizer.fit_transform(df['Testo_Pulito'])
y = df['Codice Lingua']
print(f"Matrice features: {X.shape[0]} documenti x {X.shape[1]} trigrammi")

# Train/test split con stratificazione
print("\nSplit train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)
print(f"Train: {X_train.shape[0]} esempi, Test: {X_test.shape[0]} esempi")

# Training del modello
print("\nTraining Multinomial Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Valutazione
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\nAccuratezza Train: {acc_train*100:.2f}%")
print(f"Accuratezza Test: {acc_test*100:.2f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Tedesco', 'Inglese', 'Italiano']))

# Salvataggio modello e vectorizer
print("\nSalvataggio modello...")
with open('language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modello salvato: language_model.pkl")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Vectorizer salvato: vectorizer.pkl")

print("\n✓ Training completato con successo!")