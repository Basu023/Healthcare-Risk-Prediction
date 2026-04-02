import numpy as np
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import google.generativeai as genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Health Risk Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.prediction-box {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: bold;
}
.disease { background-color: #ffebee; color: #c62828; }
.healthy { background-color: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(name):
    path = os.path.join(BASE_DIR, 'saved_models', name)
    with open(path, 'rb') as f:
        return pickle.load(f)

diabetes_model = load_model('diabetes_model.sav')
heart_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')

# ---------------- Gemini FUNCTION ----------------

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text
        else:
            return "⚠️ No response from AI"

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        'Healthcare Risk Prediction Dashboard',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Chatbot'],
        icons=['activity', 'heart', 'person', 'chat'],
        default_index=0
    )
    st.markdown("---")
    st.caption("This is for educational purposes only")

# ---------------- RESULT FUNCTION ----------------
def show_result(prediction, proba, disease_name):
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.markdown(f'<div class="prediction-box disease">{disease_name} RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box healthy">HEALTHY</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{max(proba)*100:.1f}%")

    with col3:
        risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
        st.metric("Risk Level", risk)

    fig = go.Figure(data=[
        go.Bar(name='Healthy', x=['Probability'], y=[proba[0]]),
        go.Bar(name='Disease', x=['Probability'], y=[proba[1]])
    ])
    fig.update_layout(barmode='group', height=300)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UNIVERSAL PRED FUNCTION ----------------
def get_prediction_and_proba(model, user_input):
    prediction = model.predict([user_input])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([user_input])[0]
    else:
        score = model.decision_function([user_input])[0]
        prob = 1 / (1 + np.exp(-score))
        proba = [1 - prob, prob]

    return prediction, proba

# ---------------- DIABETES ----------------
if selected == 'Diabetes Prediction':

    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.slider('Pregnancies', 0, 20, 1)
        SkinThickness = st.number_input('Skin Thickness', 0.0)
        DPF = st.number_input('Diabetes Pedigree Function', 0.0)

    with col2:
        Glucose = st.number_input('Glucose Level', 0.0)
        Insulin = st.number_input('Insulin Level', 0.0)
        Age = st.slider('Age', 0, 100, 25)

    with col3:
        BloodPressure = st.number_input('Blood Pressure', 0.0)
        BMI = st.number_input('BMI', 0.0)

    if st.button("Diabetes Test Result"):

        user_input = [Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI, DPF, Age]

        prediction, proba = get_prediction_and_proba(diabetes_model, user_input)

        st.session_state.user_data = {
            "disease": "diabetes",
            "inputs": {
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DPF": DPF,
                "Age": Age
            },
            "prediction": int(prediction),
            "confidence": float(max(proba))
        }
        st.session_state.last_result = "diabetes"

        show_result(prediction, proba, "DIABETES")
        st.subheader("🔍 Debug / Stored User Data")

        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# ---------------- HEART ----------------
if selected == 'Heart Disease Prediction':

    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', 0, 100, 30)
        trestbps = st.number_input('Resting BP', 0.0)
        restecg = st.number_input('Rest ECG', 0.0)
        oldpeak = st.number_input('Oldpeak', 0.0)
        ca = st.number_input('CA', 0.0)

    with col2:
        sex = st.slider('Sex (0=F,1=M)', 0, 1, 1)
        chol = st.number_input('Cholesterol', 0.0)
        thalach = st.number_input('Max Heart Rate', 0.0)
        slope = st.number_input('Slope', 0.0)
        thal = st.number_input('Thal', 0.0)

    with col3:
        cp = st.number_input('Chest Pain Type', 0.0)
        fbs = st.slider('FBS >120 (0/1)', 0, 1, 0)
        exang = st.slider('Exercise Angina (0/1)', 0, 1, 0)

    if st.button("Heart Disease Test Result"):

        user_input = [age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]

        prediction, proba = get_prediction_and_proba(heart_model, user_input)

        st.session_state.user_data = {
            "disease": "heart",
            "inputs": {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            },
            "prediction": int(prediction),
            "confidence": float(max(proba))
        }

        st.session_state.last_result = "heart"

        show_result(prediction, proba, "HEART DISEASE")
        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# ---------------- PARKINSONS ----------------
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('Fo', 0.0)
        RAP = st.number_input('RAP', 0.0)
        APQ3 = st.number_input('APQ3', 0.0)
        NHR = st.number_input('NHR', 0.0)
        spread1 = st.number_input('spread1', 0.0)

    with col2:
        fhi = st.number_input('Fhi', 0.0)
        PPQ = st.number_input('PPQ', 0.0)
        APQ5 = st.number_input('APQ5', 0.0)
        HNR = st.number_input('HNR', 0.0)
        spread2 = st.number_input('spread2', 0.0)

    with col3:
        flo = st.number_input('Flo', 0.0)
        DDP = st.number_input('DDP', 0.0)
        APQ = st.number_input('APQ', 0.0)
        RPDE = st.number_input('RPDE', 0.0)
        D2 = st.number_input('D2', 0.0)

    with col4:
        Jitter_percent = st.number_input('Jitter %', 0.0)
        Shimmer = st.number_input('Shimmer', 0.0)
        DDA = st.number_input('DDA', 0.0)
        DFA = st.number_input('DFA', 0.0)

    with col5:
        Jitter_Abs = st.number_input('Jitter Abs', 0.0)
        Shimmer_dB = st.number_input('Shimmer dB', 0.0)
        PPE = st.number_input('PPE', 0.0)

    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                      APQ3, APQ5, APQ, DDA, NHR, HNR,
                      RPDE, DFA, spread1, spread2, D2, PPE]

        prediction, proba = get_prediction_and_proba(parkinsons_model, user_input)

        st.session_state.user_data = {
            "disease": "parkinsons",
            "inputs": {
                "fo": fo,
                "fhi": fhi,
                "flo": flo,
                "Jitter_percent": Jitter_percent,
                "Jitter_Abs": Jitter_Abs,
                "RAP": RAP,
                "PPQ": PPQ,
                "DDP": DDP,
                "Shimmer": Shimmer,
                "Shimmer_dB": Shimmer_dB,
                "APQ3": APQ3,
                "APQ5": APQ5,
                "APQ": APQ,
                "DDA": DDA,
                "NHR": NHR,
                "HNR": HNR,
                "RPDE": RPDE,
                "DFA": DFA,
                "spread1": spread1,
                "spread2": spread2,
                "D2": D2,
                "PPE": PPE
            },
            "prediction": int(prediction),
            "confidence": float(max(proba))
        }

        st.session_state.last_result = "parkinsons"

        show_result(prediction, proba, "PARKINSONS")
        
        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# ---------------- CHATBOT ----------------
if selected == "Chatbot":

    st.title("💬 Health Chatbot")
    st.caption("Powered by Gemini AI")

    if "user_data" in st.session_state:
        st.info(f"Using context: {st.session_state.user_data['disease']}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your health question")

    if st.button("Send"):

        if not user_input:
            st.warning("Please enter a question")
        else:
            st.session_state.chat_history.append(("You", user_input))

            user_data = st.session_state.get("user_data", None)

            if user_data:
                prompt = f"""
                You are a medical assistant.

                User condition: {user_data['disease']}
                Prediction result: {"High Risk" if user_data['prediction']==1 else "Low Risk"}
                Confidence: {user_data['confidence']}

                User health data:
                {user_data['inputs']}

                User question: {user_input}

                Give:
                1. Simple explanation
                2. Risk interpretation
                3. Practical advice (diet, lifestyle)

                Keep it short and clear.
                """
            else:
                prompt = f"You are a helpful health assistant. Answer: {user_input}"

            with st.spinner("Thinking..."):
                reply = ask_gemini(prompt)

            st.session_state.chat_history.append(("Bot", reply))

    st.markdown("---")

    chat_box = st.container(height=400)

    with chat_box:
        for role, msg in st.session_state.chat_history[-10:]:
            st.write(f"**{role}:** {msg}")