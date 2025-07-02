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
    with open('model_svm.pkl', 'rb') as f:
        best_svm_model = pickle.load(f)
    with open('model_dt.pkl', 'rb') as f:
        best_dt_model = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f:
        best_rf_model = pickle.load(f)
    with open('model_xgb.pkl', 'rb') as f:
        best_xgb_model = pickle.load(f)
except FileNotFoundError:
    st.error("Pastikan file model (.pkl) tersedia di direktori yang sama.")
    st.stop()

# Load encoders and scaler
try:
    with open('3Gene.pkl', 'rb') as f:
        label_encoder_3Gene = pickle.load(f)
    with open('CancerType.pkl', 'rb') as f:
        label_encoder_cancer_type = pickle.load(f)
    with open('Cellularity.pkl', 'rb') as f:
        label_encoder_cellularity = pickle.load(f)
    with open('Chemotherapy.pkl', 'rb') as f:
        label_encoder_chemotherapy = pickle.load(f)
    with open('ERStatus.pkl', 'rb') as f:
        label_encoder_er_status = pickle.load(f)
    with open('ERstatusbyIHC.pkl', 'rb') as f:
        label_encoder_er_status_by_ihc = pickle.load(f)
    with open('HER2Status.pkl', 'rb') as f:
        label_encoder_her2_status = pickle.load(f)
    with open('HER2statusbySNP6.pkl', 'rb') as f:
        label_encoder_her2_status_by_snp6 = pickle.load(f)
    with open('HormoneTherapy.pkl', 'rb') as f:
        label_encoder_hormone_therapy = pickle.load(f)
    with open('InferredMenopausal.pkl', 'rb') as f:
        label_encoder_inferred_menopausal = pickle.load(f)
    with open('IntegrativeCluster.pkl', 'rb') as f:
        label_encoder_intergrative_cluster = pickle.load(f)
    with open('PRStatus.pkl', 'rb') as f:
        label_encoder_pr_status = pickle.load(f)
    with open('Pam50.pkl', 'rb') as f:
        label_encoder_pam50 = pickle.load(f)
    with open('PrimaryTumor.pkl', 'rb') as f:
        label_encoder_primary_tumor = pickle.load(f)
    with open('RadioTherapy.pkl', 'rb') as f:
        label_encoder_radio_therapy = pickle.load(f)
    with open('RelapseFree.pkl', 'rb') as f:
        label_encoder_relapse_free = pickle.load(f)
    with open('TumorOther.pkl', 'rb') as f:
        label_encoder_tumor_other = pickle.load(f)
    with open('TypeofBreastSurgery.pkl', 'rb') as f:
        label_encoder_type_of_breast_surgery = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
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
    processed_df['Type of Breast Surgery'] = label_encoder_type_of_breast_surgery.transform(processed_df['Type of Breast Surgery'])
    processed_df['Cancer Type Detailed'] = label_encoder_cancer_type.transform(processed_df['Cancer Type Detailed'])
    processed_df['Cellularity'] = label_encoder_cellularity.transform(processed_df['Cellularity'])
    processed_df['Chemotherapy'] = label_encoder_chemotherapy.transform(processed_df['Chemotherapy'])
    processed_df['Pam50 + Claudin-low subtype'] = label_encoder_pam50.transform(processed_df['Pam50 + Claudin-low subtype'])
    processed_df['ER status measured by IHC'] = label_encoder_er_status_by_ihc.transform(processed_df['ER status measured by IHC'])
    processed_df['ER Status'] = label_encoder_er_status.transform(processed_df['ER Status'])
    processed_df['HER2 status measured by SNP6'] = label_encoder_her2_status_by_snp6.transform(processed_df['HER2 status measured by SNP6'])
    processed_df['HER2 Status'] = label_encoder_her2_status.transform(processed_df['HER2 Status'])
    processed_df['Tumor Other Histologic Subtype'] = label_encoder_tumor_other.transform(processed_df['Tumor Other Histologic Subtype'])
    processed_df['Hormone Therapy'] = label_encoder_hormone_therapy.transform(processed_df['Hormone Therapy'])
    processed_df['Inferred Menopausal State'] = label_encoder_inferred_menopausal.transform(processed_df['Inferred Menopausal State'])
    processed_df['Integrative Cluster'] = label_encoder_intergrative_cluster.transform(processed_df['Integrative Cluster'])
    processed_df['Primary Tumor Laterality'] = label_encoder_primary_tumor.transform(processed_df['Primary Tumor Laterality'])
    processed_df['PR Status'] = label_encoder_pr_status.transform(processed_df['PR Status'])
    processed_df['Radio Therapy'] = label_encoder_radio_therapy.transform(processed_df['Radio Therapy'])
    processed_df['Relapse Free Status'] = label_encoder_relapse_free.transform(processed_df['Relapse Free Status'])
    processed_df['3-Gene classifier subtype'] = label_encoder_3Gene.transform(processed_df['3-Gene classifier subtype'])
    
    return processed_df

def prediction_page():
    st.title("Prediksi Kelangsungan Hidup Pasien Kanker Payudara")
    st.write("Masukkan data pasien di bawah ini:")

    inputs = {}
    
    inputs['Age at Diagnosis'] = st.number_input("Age at Diagnosis", min_value=20.0, max_value=100.0, value=60.0, step=1.0)
    inputs['Type of Breast Surgery'] = st.selectbox(
        "Breast Surgery Type", ['Mastectomy', 'Breast Conserving']
    )
    inputs['Cancer Type Detailed'] = st.selectbox(
        "Cancer Type Detailed", ['Breast Invasive Ductal Carcinoma', 'Breast Mixed Ductal and Lobular Carcinoma', 'Breast Invasive Lobular Carcinoma', 'Invasive Breast Carcinoma', 'Breast Invasive Mixed Mucinous Carcinoma', 'Breast Angiosarcoma', 'Breast', 'Metaplastic Breast Cancer']
    )
    inputs['Cellularity'] = st.selectbox("Cellularity", ['High', 'Moderate', 'Low'])
    inputs['Chemotherapy'] = st.selectbox("Chemotherapy", ['No', 'Yes'])
    inputs['Pam50 + Claudin-low subtype'] = st.selectbox(
        "PAM50 Subtype", ['claudin-low', 'LumA', 'LumB', 'Normal', 'Her2', 'Basal', 'NC']
    )
    inputs['Cohort'] = st.number_input("Cohort", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    inputs['ER status measured by IHC'] = st.selectbox(
        "ER status measured by IHC", ['Positve', 'Negative']
    )
    inputs['ER Status'] = st.selectbox("ER Status", ['Positive', 'Negative'])
    inputs['Neoplasm Histologic Grade'] = st.number_input("Neoplasm Histologic Grade", min_value=1.0, max_value=5.0, value=2.0, step=1.0)
    inputs['HER2 status measured by SNP6'] = st.selectbox(
        "HER2 status measured by SNP6", ['Neutral', 'Loss', 'Gain', 'Undef']
    )
    inputs['HER2 Status'] = st.selectbox("HER2 Status", ['Negative', 'Positive'])
    inputs['Tumor Other Histologic Subtype'] = st.selectbox(
        "Other Histologic Subtype", ['Ductal/NST', 'Mixed', 'Lobular', 'Tubular/ cribriform', 'Mucinous', 'Medullary', 'Other', 'Metaplastic']
    )
    inputs['Hormone Therapy'] = st.selectbox("Hormone Therapy", ['Yes', 'No'])
    inputs['Inferred Menopausal State'] = st.selectbox(
        "Menopausal State", ['Post', 'Pre']
    )
    inputs['Integrative Cluster'] = st.selectbox(
        "Integrative Cluster", ['4ER+', '3', '9', '7', '4ER-', '5', '8', '10', '1', '2', '6']
    )
    inputs['Primary Tumor Laterality'] = st.selectbox(
        "Primary Tumor Laterality", ['Right', 'Left']
    )
    inputs['Lymph nodes examined positive'] = st.number_input("Positive Lymph Nodes", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
    inputs['Mutation Count'] = st.number_input("Mutation Count", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    inputs['Nottingham prognostic index'] = st.number_input("Nottingham Prognostic Index", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    inputs['Overall Survival (Months)'] = st.number_input("Overall Survival (Months)", min_value=0, max_value=400, value=100, step=1)
    inputs['PR Status'] = st.selectbox("PR Status", ['Negative', 'Positive'])
    inputs['Radio Therapy'] = st.selectbox("Radio Therapy", ['Yes', 'No'])
    inputs['Relapse Free Status'] = st.selectbox(
        "Relapse Free Status", ['Not Recurred', 'Recurred']
    )
    inputs['3-Gene classifier subtype'] = st.selectbox(
        "3-Gene Subtype", ['ER-/HER2-', 'ER+/HER2- High Prolif', 'ER+/HER2- Low Prolif', 'HER2+']
    )
    inputs['Tumor Size'] = st.number_input("Tumor Size (mm)", min_value=1.0, max_value=200.0, value=30.0, step=1.0)
    inputs['Tumor Stage'] = st.number_input("Tumor Stage", min_value=0.0, max_value=4.0, value=1.0, step=1.0)
    
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

        if prediction[0] == 0:
            result = "Living or Died of Other Causes"
        else:
            result = "Died of Breast Cancer"
            
        st.markdown(f"""
        <div style="background-color:#e6f7ff; margin-bottom:10px; padding:20px; border-radius:10px; border: 1px solid #91d5ff">
            <h3 style="color:#0050b3;">Hasil Prediksi Model {selected_model}</h3>
            <p style="font-size:20px;"><strong>{result}</strong></p>
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
