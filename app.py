import numpy as np
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import google.generativeai as genai

from auth import (
    create_users_table,
    signup_user,
    login_user
)

# PAGE CONFIG 
st.set_page_config(page_title="Health Risk Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

# STYLE
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

# LOAD MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(name):
    path = os.path.join(BASE_DIR, 'saved_models', name)
    with open(path, 'rb') as f:
        return pickle.load(f)

diabetes_model = load_model('diabetes_model.sav')
heart_model = load_model('heart_disease_model.sav')
heart_scaler = load_model('heart_scaler.sav')
parkinsons_model = load_model('parkinsons_model.sav')

create_users_table()

# Gemini FUNCTION

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

# SIDEBAR
with st.sidebar:

    if st.session_state.logged_in:

        st.success(
            f"👋 Welcome, {st.session_state.username}"
        )

        selected = option_menu(
            'Healthcare Risk Prediction Dashboard',
            ['Diabetes Prediction',
             'Heart Disease Prediction',
             'Parkinsons Prediction',
             'Chatbot'],
            icons=['activity', 'heart', 'person', 'chat'],
            default_index=0
        )

        st.markdown("---")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

        st.caption("This is for educational purposes only")

    else:
        selected = None


# LOGIN / SIGNUP
if not st.session_state.logged_in:

    st.title("🔐 Healthcare Login System")

    menu = st.selectbox(
        "Select Option",
        ["Login", "Signup"]
    )

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if menu == "Signup":

        email = st.text_input("Email")

        if st.button("Create Account"):

            success = signup_user(
                username,
                email,
                password
            )

            if success:
                st.success("Account created successfully!")
            else:
                st.error("Username or Email already exists")

    elif menu == "Login":

        if st.button("Login"):

            if login_user(username, password):

                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()

            else:
                st.error("Invalid username or password")

    st.stop()
    
# RESULT FUNCTION
def show_result(prediction, proba, disease_name, disease_class=1):
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == disease_class:
            st.markdown(
                f'<div class="prediction-box disease">{disease_name} RISK</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box healthy">HEALTHY</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.metric("Confidence", f"{max(proba)*100:.1f}%")

    with col3:
        disease_prob = proba[disease_class]

        risk = (
            "High" if disease_prob > 0.7
            else "Medium" if disease_prob > 0.4
            else "Low"
        )

        st.metric("Risk Level", risk)

    fig = go.Figure(data=[
        go.Bar(name='Healthy', x=['Probability'], y=[proba[0]]),
        go.Bar(name='Disease', x=['Probability'], y=[proba[1]])
    ])
    fig.update_layout(barmode='group', height=300)
    st.plotly_chart(fig, use_container_width=True)

# UNIVERSAL PRED FUNCTION
def get_prediction_and_proba(model, user_input):
    prediction = model.predict([user_input])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([user_input])[0]
    else:
        score = model.decision_function([user_input])[0]
        prob = 1 / (1 + np.exp(-score))
        proba = [1 - prob, prob]

    return prediction, proba

# DIABETES
if selected == 'Diabetes Prediction':

    saved_diabetes = st.session_state.form_data.get(
        "diabetes", {}
    )

    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.slider(
            'Pregnancies',
            0, 20,
            saved_diabetes.get("Pregnancies", 0),
            key="diabetes_pregnancies"
        )

        SkinThickness = st.number_input(
            'Skin Thickness (mm)',
            min_value=0.0,
            value=saved_diabetes.get("SkinThickness", 0.0),
            help='Triceps skin fold thickness measured in mm',
            key="diabetes_skin"
        )

        DPF = st.number_input(
            'Diabetes Pedigree Function (Score)',
            min_value=0.0,
            value=saved_diabetes.get("DPF", 0.0),
            help='Indicates diabetes likelihood based on family history',
            key="diabetes_dpf"
        )

    with col2:
        Glucose = st.number_input(
            'Glucose Level (mg/dL)',
            min_value=0.0,
            value=saved_diabetes.get("Glucose", 0.0),
            help='Blood glucose concentration',
            key="diabetes_glucose"
        )

        Insulin = st.number_input(
            'Insulin Level (IU/mL)',
            min_value=0.0,
            value=saved_diabetes.get("Insulin", 0.0),
            help='Insulin level in blood',
            key="diabetes_insulin"
            
        )

        Age = st.slider(
            'Age (Years)',
            0, 100,
            saved_diabetes.get("Age", 0),
            key="diabetes_age"
        )

    with col3:
        BloodPressure = st.number_input(
            'Blood Pressure (mmHg)',
            min_value=0.0,
            value=saved_diabetes.get("BloodPressure", 0.0),
            help='Diastolic blood pressure',
            key="diabetes_bp"
        )

        BMI = st.number_input(
            'BMI (kg/m²)',
            min_value=0.0,
            value=saved_diabetes.get("BMI", 0.0),
            help='Body Mass Index',
            key="diabetes_bmi"
        )

    if st.button("Diabetes Test Result"):

        if (
            Glucose == 0 or
            BloodPressure == 0 or
            SkinThickness == 0 or
            Insulin == 0 or
            BMI == 0
        ):
            st.warning("⚠️ Please fill all required fields.")
            st.stop()

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

        st.session_state.form_data["diabetes"] = {
            "Pregnancies": Pregnancies,
            "SkinThickness": SkinThickness,
            "DPF": DPF,
            "Glucose": Glucose,
            "Insulin": Insulin,
            "Age": Age,
            "BloodPressure": BloodPressure,
            "BMI": BMI
}

        show_result(prediction, proba, "DIABETES", disease_class=1)
        st.subheader("🔍 Debug / Stored User Data")

        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# HEART
if selected == 'Heart Disease Prediction':

    saved_heart = st.session_state.form_data.get(
        "heart", {}
    )

    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            'Age (Years)',
            0, 100,
            saved_heart.get("age", 30),
            key="heart_age"
        )

        trestbps = st.number_input(
            'Resting Blood Pressure (mmHg)',
            min_value=0.0,
            value=saved_heart.get("trestbps", 0.0),
            help='Resting blood pressure in mmHg',
            key="heart_trestbps"
        )

        restecg = st.number_input(
            'Rest ECG (0-2)',
            min_value=0.0,
            value=saved_heart.get("restecg", 0.0),
            help='''
    0 = Normal ECG  
    1 = ST-T wave abnormality  
    2 = Left ventricular hypertrophy
    ''',
            key="heart_restecg"
        )

        oldpeak = st.number_input(
            'Oldpeak (ST Depression)',
            min_value=0.0,
            value=saved_heart.get("oldpeak", 0.0),
            help='ST depression induced by exercise',
            key="heart_oldpeak"
        )

        ca = st.number_input(
            'CA (Major Vessels 0-4)',
            min_value=0.0,
            value=saved_heart.get("ca", 0.0),
            help='Number of major vessels colored by fluoroscopy',
            key="heart_ca"
        )

    with col2:
        sex = st.slider(
            'Sex (0 = Female, 1 = Male)',
            0, 1,
            saved_heart.get("sex", 1),
            key="heart_sex"
        )

        chol = st.number_input(
            'Cholesterol (mg/dL)',
            min_value=0.0,
            value=saved_heart.get("chol", 0.0),
            help='Serum cholesterol level',
            key="heart_chol"
        )

        thalach = st.number_input(
            'Max Heart Rate (bpm)',
            min_value=0.0,
            value=saved_heart.get("thalach", 0.0),
            help='Maximum heart rate achieved',
            key="heart_thalach"
        )

        slope = st.number_input(
            'Slope (0-2)',
            min_value=0.0,
            value=saved_heart.get("slope", 0.0),
            help='''
    0 = Upsloping  
    1 = Flat  
    2 = Downsloping
    ''',
            key="heart_slope"
        )

        thal = st.number_input(
            'Thal (0-3)',
            min_value=0.0,
            value=saved_heart.get("thal", 0.0),
            help='''
    0 = Normal  
    1 = Fixed Defect  
    2 = Reversible Defect  
    3 = Unknown
    ''',
            key="heart_thal"
        )

    with col3:
        cp = st.number_input(
            'Chest Pain Type (0-3)',
            min_value=0.0,
            value=saved_heart.get("cp", 0.0),
            help='''
    0 = Typical Angina  
    1 = Atypical Angina  
    2 = Non-anginal Pain  
    3 = Asymptomatic
    ''',
            key="heart_cp"
        )

        fbs = st.slider(
            'Fasting Blood Sugar >120 mg/dL (0/1)',
            0, 1,
            saved_heart.get("fbs", 0),
            help='''
    0 = No  
    1 = Yes
    ''',
            key="heart_fbs"
        )

        exang = st.slider(
            'Exercise Induced Angina (0/1)',
            0, 1,
            saved_heart.get("exang", 0),
            help='''
    0 = No  
    1 = Yes
    ''',
            key="heart_exang"
        )

    if st.button("Heart Disease Test Result"):

        if (
            trestbps == 0 or
            chol == 0 or
            thalach == 0
        ):
            st.warning("⚠️ Please fill all required fields.")
            st.stop()

        user_input = [age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]

        scaled_input = heart_scaler.transform([user_input])

        prediction = heart_model.predict(scaled_input)[0]
        proba = heart_model.predict_proba(scaled_input)[0]

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

        st.session_state.form_data["heart"] = {
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
    }

        show_result(prediction, proba, "HEART DISEASE", disease_class=0)
        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# PARKINSONS
if selected == "Parkinsons Prediction":

    saved_park = st.session_state.form_data.get(
        "parkinsons", {}
    )

    st.title("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input(
            'Fo - Fundamental Frequency (Hz)',
            min_value=0.0,
            value=saved_park.get("fo", 0.0),
            help='Average vocal fundamental frequency',
            key="park_fo"
        )

        RAP = st.number_input(
            'RAP (%)',
            min_value=0.0,
            value=saved_park.get("RAP", 0.0),
            help='Relative Amplitude Perturbation in voice',
            key="park_rap"
        )

        APQ3 = st.number_input(
            'APQ3 (%)',
            min_value=0.0,
            value=saved_park.get("APQ3", 0.0),
            help='Amplitude Perturbation Quotient (3-point)',
            key="park_apq3"
        )

        NHR = st.number_input(
            'NHR (Ratio)',
            min_value=0.0,
            value=saved_park.get("NHR", 0.0),
            help='Noise-to-Harmonics Ratio',
            key="park_nhr"
        )

        spread1 = st.number_input(
            'Spread1',
            value=saved_park.get("spread1", 0.0),
            help='Nonlinear measure of voice frequency variation',
            key="park_spread1"
        )

    with col2:
        fhi = st.number_input(
            'Fhi - Highest Frequency (Hz)',
            min_value=0.0,
            value=saved_park.get("fhi", 0.0),
            help='Highest vocal fundamental frequency',
            key="park_fhi"
        )

        PPQ = st.number_input(
            'PPQ (%)',
            min_value=0.0,
            value=saved_park.get("PPQ", 0.0),
            help='Pitch Period Perturbation Quotient',
            key="park_ppq"
        )

        APQ5 = st.number_input(
            'APQ5 (%)',
            min_value=0.0,
            value=saved_park.get("APQ5", 0.0),
            help='Amplitude Perturbation Quotient (5-point)',
            key="park_apq5"
        )

        HNR = st.number_input(
            'HNR (dB)',
            min_value=0.0,
            value=saved_park.get("HNR", 0.0),
            help='Harmonics-to-Noise Ratio',
            key="park_hnr"
        )

        spread2 = st.number_input(
            'Spread2',
            min_value=0.0,
            value=saved_park.get("spread2", 0.0),
            help='Nonlinear measure of voice frequency variation',
            key="park_spread2"
        )

    with col3:
        flo = st.number_input(
            'Flo - Lowest Frequency (Hz)',
            min_value=0.0,
            value=saved_park.get("flo", 0.0),
            help='Lowest vocal fundamental frequency',
            key="park_flo"
        )

        DDP = st.number_input(
            'DDP (%)',
            min_value=0.0,
            value=saved_park.get("DDP", 0.0),
            help='Average absolute difference of jitter',
            key="park_ddp"
        )

        APQ = st.number_input(
            'APQ (%)',
            min_value=0.0,
            value=saved_park.get("APQ", 0.0),
            help='Amplitude Perturbation Quotient',
            key="park_apq"
        )

        RPDE = st.number_input(
            'RPDE',
            min_value=0.0,
            value=saved_park.get("RPDE", 0.0),
            help='Recurrence Period Density Entropy (voice signal complexity)',
            key="park_rpde"
        )

        D2 = st.number_input(
            'D2',
            min_value=0.0,
            value=saved_park.get("D2", 0.0),
            help='Correlation dimension for voice dynamics',
            key="park_d2"
        )

    with col4:
        Jitter_percent = st.number_input(
            'Jitter (%)',
            min_value=0.0,
            value=saved_park.get("Jitter", 0.0),
            help='Variation in vocal frequency',
            key="park_jitter_percent"
        )

        Shimmer = st.number_input(
            'Shimmer (dB)',
            min_value=0.0,
            value=saved_park.get("Shimmer", 0.0),
            help='Variation in vocal amplitude',
            key="park_shimmer"
        )

        DDA = st.number_input(
            'DDA',
            min_value=0.0,
            value=saved_park.get("DDA", 0.0),
            help='Average absolute difference of amplitudes',
            key="park_dda"
        )

        DFA = st.number_input(
            'DFA',
            min_value=0.0,
            value=saved_park.get("DFA", 0.0),
            help='Detrended Fluctuation Analysis of voice signal',
            key="park_dfa"
        )
    with col5:
        Jitter_Abs = st.number_input(
            'Jitter Absolute (ms)',
            min_value=0.0,
            value=saved_park.get("Jitter_Abs", 0.0),
            help='Absolute variation in vocal frequency',
            key="park_jitter_abs"
        )

        Shimmer_dB = st.number_input(
            'Shimmer (dB)',
            min_value=0.0,
            value=saved_park.get("Shimmer_dB", 0.0),
            help='Variation in vocal amplitude measured in decibels',
            key="park_shimmer_db"
        )

        PPE = st.number_input(
            'PPE',
            min_value=0.0,
            value=saved_park.get("PPE", 0.0),
            help='Pitch Period Entropy - measures irregularities in voice',
            key="park_ppe"
        )

    if st.button("Parkinson's Test Result"):

        if (
            fo == 0 or
            fhi == 0 or
            flo == 0
        ):
            st.warning("⚠️ Please fill all required fields.")
            st.stop()

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

        st.session_state.form_data["parkinsons"] = {
            "fo": fo,
            "fhi": fhi,
            "flo": flo,
            "RAP": RAP,
            "PPQ": PPQ,
            "DDP": DDP,
            "APQ3": APQ3,
            "APQ5": APQ5,
            "APQ": APQ,
            "NHR": NHR,
            "HNR": HNR,
            "spread1": spread1,
            "spread2": spread2,
            "D2": D2,
            "PPE": PPE,
            "RPDE": RPDE,
            "DDA": DDA,
            "DFA": DFA,
            "Jitter": Jitter_percent,
            "Jitter_Abs": Jitter_Abs,
            "Shimmer": Shimmer,
            "Shimmer_dB": Shimmer_dB
        }

        show_result(prediction, proba, "PARKINSONS", disease_class=1)
        
        with st.expander("📊 View Stored User Data"):
            st.json(st.session_state.user_data)

# CHATBOT
if selected == "Chatbot":

    st.title("💬 Health Chatbot")
    st.caption("Powered by Gemini AI")

    if "user_data" in st.session_state:
        st.info(f"Using context: {st.session_state.user_data['disease']}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your health question",key="chat_input")

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