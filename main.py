import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    with open('label_encoder_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
except FileNotFoundError:
    st.error("Pastikan file encoder (.pkl) tersedia.")
    st.stop()

# Define columns used in training (based on your notebook)
feature_columns_order = [
    'Age', 'Gender', 'Protein1', 'Protein2', 'Protein3', 'Protein4',
    'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status',
    'Surgery_type', 'Patient_Status' # Include target temporarily to define structure
]

# This order must match the order of features the trained models expect
processed_feature_columns_order = [
    'Age', 'Gender', 'Protein1', 'Protein2', 'Protein3', 'Protein4', 'Tumour_Stage',
    'Histology_Infiltrating Ductal Carcinoma', 'Histology_Infiltrating Lobular Carcinoma', 'Histology_Mucinous Carcinoma',
    'ER status', 'PR status', 'HER2 status',
    'Surgery_type_Lumpectomy', 'Surgery_type_Modified Radical Mastectomy', 'Surgery_type_Other', 'Surgery_type_Simple Mastectomy'
]

def preprocess_input(input_df):
    """Applies the same preprocessing steps as used during training."""
    processed_df = input_df.copy()

    # Apply Label Encoding using the loaded encoder

    processed_df['Gender'] = label_encoder_gender.transform(processed_df['Gender'])
    processed_df['Tumour_Stage'] = label_encoder_tumour_stage.transform(processed_df['Tumour_Stage'])
    processed_df['ER status'] = label_encoder_er_status.transform(processed_df['ER status'])
    processed_df['PR status'] = label_encoder_pr_status.transform(processed_df['PR status'])
    processed_df['HER2 status'] = label_encoder_her2_status.transform(processed_df['HER2 status'])
    
    # Apply One-Hot Encoding using the loaded encoder
    columns_to_onehot_encode = ['Histology', 'Surgery_type'] # Sesuai notebook Anda

    # Transform the relevant columns
    encoded_features = onehot_encoder.transform(processed_df[columns_to_onehot_encode]).toarray()

    # Create a DataFrame with encoded features
    # Get feature names from the loaded onehot_encoder
    encoded_feature_names = onehot_encoder.get_feature_names_out(columns_to_onehot_encode)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=processed_df.index)

    # Drop original One-Hot encoded columns
    processed_df = processed_df.drop(columns=columns_to_onehot_encode)

    # Concatenate processed_df with encoded_df
    processed_df = pd.concat([processed_df, encoded_df], axis=1)


    # Ensure column order matches the training data
    # This list must be exactly the same as the columns used to train your model,
    # in the exact order, AFTER preprocessing.
    # Based on your notebook, this should include the original numerical/label encoded
    # columns plus the new one-hot encoded columns.

    # Recreate this list based on your preprocessed training data's columns
    processed_feature_columns_order = [
        'Age', 'Gender', 'Protein1', 'Protein2', 'Protein3', 'Protein4', 'Tumour_Stage',
        'ER status', 'PR status', 'HER2 status',
        'Histology_Infiltrating Ductal Carcinoma', 'Histology_Infiltrating Lobular Carcinoma', 'Histology_Mucinous Carcinoma',
        'Surgery_type_Lumpectomy', 'Surgery_type_Modified Radical Mastectomy', 'Surgery_type_Other', 'Surgery_type_Simple Mastectomy'
    ]

    # Filter and reorder columns to match processed_feature_columns_order
    # Use .reindex(columns=...) to handle potential missing columns (shouldn't happen if preprocessing is consistent)
    processed_df = processed_df.reindex(columns=processed_feature_columns_order, fill_value=0)


    return processed_df

def prediction_page():
    st.title("Prediksi Kelangsungan Hidup Pasien Kanker Payudara")
    st.write("Masukkan data pasien di bawah ini:")

    # Input fields for features
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", ["FEMALE", "MALE"])
    protein1 = st.number_input("Protein1", value=1.0)
    protein2 = st.number_input("Protein2", value=1.0)
    protein3 = st.number_input("Protein3", value=1.0)
    protein4 = st.number_input("Protein4", value=1.0)
    tumour_stage = st.selectbox("Tumour Stage", ["I", "II", "III"])
    histology = st.selectbox("Histology", ["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Mucinous Carcinoma"])
    er_status = st.selectbox("ER status", ["Positive", "Negative"])
    pr_status = st.selectbox("PR status", ["Positive", "Negative"])
    her2_status = st.selectbox("HER2 status", ["Positive", "Negative"])
    surgery_type = st.selectbox("Surgery type", ["Lumpectomy", "Simple Mastectomy", "Modified Radical Mastectomy", "Other"])

    # Create a dictionary from input values
    input_data = {
        'Age': age,
        'Gender': gender,
        'Protein1': protein1,
        'Protein2': protein2,
        'Protein3': protein3,
        'Protein4': protein4,
        'Tumour_Stage': tumour_stage,
        'Histology': histology,
        'ER status': er_status,
        'PR status': pr_status,
        'HER2 status': her2_status,
        'Surgery_type': surgery_type
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # This helps in the preprocessing step.
    for col in feature_columns_order:
        if col not in input_df.columns and col != 'Patient_Status':
             # You might need to add default values based on your training data's mean/mode
             # For simplicity, we'll add them with placeholder values
            if col in ['Protein1', 'Protein2', 'Protein3', 'Protein4']:
                 input_df[col] = 0.0 # Example default for numerical
            else:
                 input_df[col] = 'Unknown' # Example default for categorical

    # Reorder columns to match the expected input format before preprocessing
    input_df = input_df[[col for col in feature_columns_order if col != 'Patient_Status']]

    # Preprocess the input data
    processed_input_df = preprocess_input(input_df)

    st.subheader("Pilih Model untuk Prediksi:")
    selected_model = st.selectbox("Model", ["SVM 82%", "Decision Tree 74%", "Random Forest 82%", "XGBoost 78%"])

    if st.button("Prediksi"):
        if selected_model == "SVM 82%":
            prediction = best_svm_model.predict(processed_input_df)
        elif selected_model == "Decision Tree 74%":
            prediction = best_dt_model.predict(processed_input_df)
        elif selected_model == "Random Forest 82%":
            prediction = best_rf_model.predict(processed_input_df)
        elif selected_model == "XGBoost 78%":
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
