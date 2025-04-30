import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement du modèle et du scaler
model = joblib.load("logreg_smote_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prédiction du Churn client")

st.markdown("Remplis les informations du client pour prédire s’il risque de résilier son abonnement.")

# Interface utilisateur (adaptée pour un modèle type Telco Churn)
gender = st.selectbox("Genre", ["Male", "Female"])
senior = st.selectbox("Client senior ?", ["No", "Yes"])
partner = st.selectbox("A un partenaire ?", ["No", "Yes"])
dependents = st.selectbox("A des personnes à charge ?", ["No", "Yes"])
tenure = st.slider("Durée d’abonnement (mois)", 0, 72, 12)
phone_service = st.selectbox("Service téléphonique ?", ["No", "Yes"])
multiple_lines = st.selectbox("Lignes multiples ?", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Type d'internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Sécurité en ligne ?", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Sauvegarde en ligne ?", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Protection d'appareil ?", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Support technique ?", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV ?", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming films ?", ["No", "Yes", "No internet service"])
contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Facturation sans papier ?", ["Yes", "No"])
payment_method = st.selectbox("Méthode de paiement", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Frais mensuels", value=70.0)
total_charges = st.number_input("Total payé", value=800.0)

# Création du DataFrame utilisateur
input_dict = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

user_df = pd.DataFrame([input_dict])

# Encodage des variables catégorielles
user_df_encoded = pd.get_dummies(user_df)

# Récupérer les colonnes attendues par le modèle
expected_columns = scaler.feature_names_in_

# Ajouter les colonnes manquantes
for col in expected_columns:
    if col not in user_df_encoded.columns:
        user_df_encoded[col] = 0

# Réordonner les colonnes
user_df_encoded = user_df_encoded[expected_columns]

# Normaliser
user_scaled = scaler.transform(user_df_encoded)

# Prédiction
if st.button(" Prédire"):
    prediction = model.predict(user_scaled)
    proba = model.predict_proba(user_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f" Risque élevé de churn : {proba*100:.2f}%")
    else:
        st.success(f" Client fidèle : {100 - proba*100:.2f}% de chances de rester")
