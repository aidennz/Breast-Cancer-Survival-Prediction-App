import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://about.canva.com/wp-content/uploads/sites/8/2019/05/light-blue.png");
    background-size: 2000px;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.1);
}

.mainbox {
    background: rgba(255,255,255,0.95);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    margin-left: -70px;
    width: 900px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.15);
}

.mainbox h1, .mainbox h2, .mainbox h3, .mainbox p, .mainbox label, .mainbox div {
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load models
try:
    with open('best_svm_model.pkl', 'rb') as f:
        best_svm_model = pickle.load(f)
    with open('best_dt_model.pkl', 'rb') as f:
        best_dt_model = pickle.load(f)
    with open('best_rf_model.pkl', 'rb') as f:
        best_rf_model = pickle.load(f)
    with open('best_xgb_model.pkl', 'rb') as f:
        best_xgb_model = pickle.load(f)
except FileNotFoundError:
    st.error("Pastikan file model (.pkl) tersedia di direktori yang sama.")
    st.stop()

# Load encoders
try:
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('label_encoder_tumour_stage.pkl', 'rb') as f:
        label_encoder_tumour_stage = pickle.load(f)
    with open('label_encoder_er_status.pkl', 'rb') as f:
        label_encoder_er_status = pickle.load(f)
    with open('label_encoder_pr_status.pkl', 'rb') as f:
        label_encoder_pr_status = pickle.load(f)
    with open('label_encoder_her2_status.pkl', 'rb') as f:
        label_encoder_her2_status = pickle.load(f)
    with open('label_encoder_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
except FileNotFoundError:
    st.error("Pastikan file encoder (.pkl) tersedia.")
    st.stop()

# Define columns used in training
numerical_features = [
    'Age at Diagnosis', 'Cohort', 'Neoplasm Histologic Grade', 'Lymph nodes examined positive', 
    'Mutation Count', 'Nottingham prognostic index',
    'Overall Survival (Months)', 'Tumor Size', 'Tumor Stage'
]

categorical_features = [
    'Type of Breast Surgery', 'Cancer Type Detailed', 'Cellularity', 'Chemotherapy',
    'Pam50 + Claudin-low subtype', 'ER status measured by IHC', 'ER Status', 'HER2 status measured by SNP6', 'HER2 Status',
    'Tumor Other Histologic Subtype', 'Hormone Therapy', 'Inferred Menopausal State', 'Integrative Cluster', 'Primary Tumor Laterality', 'PR Status',
    'Radio Therapy', 'Relapse Free Status', '3-Gene classifier subtype'
]

def preprocess_input(input_df):
    processed_df = input_df.copy()
    
    # Label encode categorical features
    for feature in categorical_features:
        if feature in label_encoders:
            processed_df[feature] = label_encoders[feature].transform(processed_df[feature])

    # Scale numerical features
    processed_df[numerical_features] = scaler.transform(processed_df[numerical_features])
    
    return processed_df

def prediction_page():
    st.title("Prediksi Kelangsungan Hidup Pasien Kanker Payudara")
    st.write("Masukkan data pasien di bawah ini:")

    inputs = {}
    
    # --- Numerical Features ---
    inputs['Age at Diagnosis'] = st.sidebar.number_input("Age at Diagnosis", min_value=20, max_value=100, value=60)
    inputs['Cohort'] = st.sidebar.number_input("Cohort", min_value=1900, max_value=2100, value=2000)
    inputs['Neoplasm Histologic Grade'] = st.sidebar.number_input("Neoplasm Histologic Grade", min_value=1, max_value=5, value=2)
    inputs['Lymph nodes examined positive'] = st.sidebar.number_input("Positive Lymph Nodes", min_value=0, max_value=50, value=0)
    inputs['Mutation Count'] = st.sidebar.number_input("Mutation Count", min_value=0, max_value=100, value=5)
    inputs['Nottingham prognostic index'] = st.sidebar.number_input("Nottingham Prognostic Index", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    inputs['Overall Survival (Months)'] = st.sidebar.number_input("Overall Survival (Months)", min_value=0, max_value=300, value=100)
    inputs['Tumor Size'] = st.sidebar.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=30)
    inputs['Tumor Stage'] = st.sidebar.number_input("Tumor Stage (angka)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # --- Categorical Features ---
    inputs['Type of Breast Surgery'] = st.sidebar.selectbox(
        "Breast Surgery Type", ['Mastectomy', 'Breast Conserving']
    )
    inputs['Cancer Type Detailed'] = st.sidebar.selectbox(
        "Cancer Type Detailed", ['Ductal/NST', 'Lobular', 'Other']
    )
    inputs['Cellularity'] = st.sidebar.selectbox("Cellularity", ['High', 'Moderate', 'Low'])
    inputs['Chemotherapy'] = st.sidebar.selectbox("Chemotherapy", ['No', 'Yes'])
    inputs['Pam50 + Claudin-low subtype'] = st.sidebar.selectbox(
        "PAM50 Subtype", ['claudin-low', 'LumA', 'LumB', 'Normal', 'Her2', 'Basal']
    )
    inputs['ER status measured by IHC'] = st.sidebar.selectbox(
        "ER status measured by IHC", ['Positive', 'Negative']
    )
    inputs['ER Status'] = st.sidebar.selectbox("ER Status", ['Positive', 'Negative'])
    inputs['HER2 status measured by SNP6'] = st.sidebar.selectbox(
        "HER2 status measured by SNP6", ['Negative', 'Positive']
    )
    inputs['HER2 Status'] = st.sidebar.selectbox("HER2 Status", ['Negative', 'Positive'])
    inputs['Tumor Other Histologic Subtype'] = st.sidebar.selectbox(
        "Other Histologic Subtype", ['None', 'Other Subtype']  # Ganti jika tahu isi yang valid
    )
    inputs['Hormone Therapy'] = st.sidebar.selectbox("Hormone Therapy", ['Yes', 'No'])
    inputs['Inferred Menopausal State'] = st.sidebar.selectbox(
        "Menopausal State", ['Post', 'Pre']
    )
    inputs['Integrative Cluster'] = st.sidebar.selectbox(
        "Integrative Cluster", ['IntClust 1', 'IntClust 2', 'IntClust 3', 'IntClust 4', 'IntClust 5', 'IntClust 6', 'IntClust 7', 'IntClust 8', 'IntClust 9', 'IntClust 10']
    )
    inputs['Primary Tumor Laterality'] = st.sidebar.selectbox(
        "Primary Tumor Laterality", ['Left', 'Right']
    )
    inputs['PR Status'] = st.sidebar.selectbox("PR Status", ['Negative', 'Positive'])
    inputs['Radio Therapy'] = st.sidebar.selectbox("Radio Therapy", ['Yes', 'No'])
    inputs['Relapse Free Status'] = st.sidebar.selectbox(
        "Relapse Free Status", ['Relapse Free Status:1', 'Relapse Free Status:0']
    )
    inputs['3-Gene classifier subtype'] = st.sidebar.selectbox(
        "3-Gene Subtype", ['ER-/HER2-', 'ER+/HER2- High Prolif', 'ER+/HER2- Low Prolif', 'HER2+']
    )

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([inputs])

    # Preprocess the input data
    processed_input_df = preprocess_input(input_df)

    st.subheader("Pilih Model untuk Prediksi:")
    selected_model = st.selectbox("Model", ["SVM 94%", "Decision Tree 91%", "Random Forest 94%", "XGBoost 93%"])

    if st.button("Prediksi"):
        if selected_model == "SVM 94%":
            prediction = best_svm_model.predict(processed_input_df)
        elif selected_model == "Decision Tree 91%":
            prediction = best_dt_model.predict(processed_input_df)
        elif selected_model == "Random Forest 94%":
            prediction = best_rf_model.predict(processed_input_df)
        elif selected_model == "XGBoost 93%":
            prediction = best_xgb_model.predict(processed_input_df)

        prediction_label = le_target.inverse_transform(prediction)[0]

        st.markdown(f"""
        <div style="background-color:#e6f7ff; margin-bottom:10px; padding:20px; border-radius:10px; border: 1px solid #91d5ff">
            <h3 style="color:#0050b3;">Hasil Prediksi Model {selected_model}</h3>
            <p style="font-size:20px;"><strong>{prediction_label}</strong></p>
        </div>
        """, unsafe_allow_html=True)


    if st.button("Kembali ke Layar Awal"):
        st.session_state.page = 'start_screen'
        st.rerun()

# --- Main application logic ---
def start_screen():
    st.markdown('''
    <div class="mainbox">
        <h1>Selamat Datang di Aplikasi Prediksi Anda</h1>
        <p>Aplikasi ini membantu memprediksi kelangsungan hidup pasien kanker payudara berdasarkan data klinis.</p>
        <p>Klik tombol di bawah untuk memulai prediksi.</p>
    ''', unsafe_allow_html=True)
    if st.button("Mulai Prediksi"):
        st.session_state.page = 'prediction_page'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'start_screen'

    # Render the current page based on session state
    if st.session_state.page == 'start_screen':
        start_screen()
    elif st.session_state.page == 'prediction_page':
        prediction_page()
