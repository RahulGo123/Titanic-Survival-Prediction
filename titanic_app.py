import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_voting_model.pkl")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# Input form
with st.form("titanic_form"):
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 29)
    sibsp = st.number_input("Number of Siblings/Spouses (SibSp)", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children (Parch)", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict Survival")

# Prediction
if submitted:
    # Create dataframe for input
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])
    
    input_data["FamilySize"] = input_data["SibSp"] + input_data["Parch"] + 1
    input_data["IsAlone"] = 1 if input_data["FamilySize"].iloc[0] == 1 else 0
    input_data["Title"] = "Mr"  # 👈 You need logic to extract from Name; 
                            # for single input you can just set default

    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ The passenger is likely to **Survive** (probability: {prob:.2f})")
    else:
        st.error(f"❌ The passenger is likely to **Not Survive** (probability: {prob:.2f})")


