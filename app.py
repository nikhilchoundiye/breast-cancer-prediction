import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import requests
import json 
# Backend API URL
API_URL = "http://127.0.0.1:8000/api/validate_license/"

# Streamlit UI for License Verification
def verify_license():
    st.title("Lab Authorization System")

    license_number = st.text_input("Enter License Number")
    expiration_date = st.date_input("License Expiration Date", min_value=datetime.today())
    if "verified" not in st.session_state:
        st.session_state.verified = False  # Initialize session state

    if st.button("Verify & Login"):
        if license_number:
            response = requests.post(API_URL, json={
                "license_number": license_number,
                "expiration_date": str(expiration_date),
            })
            
            if response.status_code == 200:
                st.success("✅ License verified! Redirecting...")
                st.session_state.verified = True  
                 # ✅ Show success message for 2 seconds before redirecting
                time.sleep(2)  
                 # ✅ Correctly update query parameters
                st.query_params = {"verified": "true"}  

                st.rerun()
            else:
                st.error("❌ License is invalid or expired.")
        else:
            st.warning("⚠️ Please enter license details.")

     # ✅ Display verified message without refreshing
    if st.session_state.verified:
        st.success("✅ Authorization Successful! You are logged in.")

def run_cancer_prediction():
    def get_min_max_values(csv_file):
    
        # Read the dataset
        df = pd.read_csv(csv_file)

        # Get min and max values for each column
        min_max_dict = {col: (df[col].min(), df[col].max()) for col in df.columns}

        return min_max_dict

    def min_max_scale(input_features, min_max_dict, feature_names):
        scaled_features = {}
        
        for i, value in enumerate(input_features):  # Loop over list
            feature = feature_names[i]  # Get feature name from list
            min_val, max_val = min_max_dict.get(feature, (0, 1))  # Get min-max for feature
            
            if max_val > min_val:  # Avoid division by zero
                scaled_features[feature] = (value - min_val) / (max_val - min_val)
            else:
                scaled_features[feature] = 0  # If min == max, set to 0 (edge case)
        
        return scaled_features




    date_of_examination = datetime.today().strftime('%d-%m-%Y')


    
    # Load trained model
    model = joblib.load("model.pkl")



    # Streamlit UI
    st.title("🔬 Breast Cancer Prediction App")
    st.write("""
    Enter Tumor Features in the Sidebar  
    This app predicts whether a tumor is Malignant (Cancerous) or Benign (Non-Cancerous)  
    Based on input tumor features, it uses a Random Forest Classifier trained on breast cancer data.
    """)

    # Sidebar for user input
    st.sidebar.header("Patient Information")

    # Input fields for patient details
    patient_name = st.sidebar.text_input("👤 Patient Name", "")
    patient_age = st.sidebar.number_input("🎂 Age", min_value=1, max_value=120, value=30, step=1)
    patient_gender = st.sidebar.selectbox("Gender", [ "Female","Male", "Other"])


    # Sidebar for user input
    st.sidebar.header("Input Tumor Features")

    #read the dataset

    # CSV Upload Feature
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

    col1, col2 = st.columns([4, 1])

    with col1:

        # List of features (Make sure to match model's feature count)
        feature_names = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
                        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]

        # Creating input fields
        input_features = []

        if uploaded_file is not None:
            # Read CSV file
            df = pd.read_csv(uploaded_file)

            # Ensure CSV contains all required features
            if all(feature in df.columns for feature in feature_names):
                st.success("✅ CSV uploaded successfully!")

                st.sidebar.write(" 🔍 Detailed View for Specific Sample")
                sample_index = st.sidebar.number_input("Enter Sample Index (Starting from 0)", min_value=0, max_value=len(df) - 1, value=0)
                
                input_features = df.iloc[sample_index].tolist()  # Take the first row's data
                max_values = df.max().to_dict()  # Get maximum values for sliders
                st.sidebar.info("Using data from uploaded CSV file.")
            else:
                st.error("❌ CSV file must contain all required features.")
        else:
            # Load dataset to set slider max values dynamically
            dataset = pd.read_csv("cleaned_breast_cancer_data.csv")  # Load original dataset
            max_values = dataset.max().to_dict()  # Maximum values for each feature
            
            for feature in feature_names:
                max_val = max_values[feature] if feature in max_values else 100.0
                value = st.sidebar.slider(f"{feature}", min_value=0.0, max_value=float(max_val), value=0.0, step=0.1)
                input_features.append(value)

        # Convert user inputs to NumPy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)

        predicted_stage = "N/A"
        stage_recommendation = "N/A"

        
        # Cancer Stage Prediction Function
        def predict_cancer_stage(radius_mean, texture_mean):
            if radius_mean > 20 and texture_mean > 20:
                return "Stage 4 (Severe)", (
                    "Stage 4 (Severe) Recommendations\n\n"
                    "Objective: Maintain quality of life, manage symptoms, and prevent severe weight loss.\n\n"
                    "Dietary Recommendations:\n"
                    "- Soft & High-Calorie, Nutrient-Dense Foods: Nut butters, avocado, soft scrambled eggs, protein-rich soups.\n"
                    "- Anti-Nausea & Digestive Comfort Foods: Ginger tea, mashed bananas, toast, applesauce.\n"
                    "- Iron & B12 for Anemia Prevention: Lean meats, fortified cereals, eggs, nutritional yeast.\n"
                    "- Bone Health & Muscle Preservation: Dairy, fortified plant milks, calcium supplements.\n"
                    "- Hydration & Electrolyte Balance: Coconut water, vegetable broths, watermelon juice.\n"
                    "- Foods to Limit: Excess salt & processed foods, red meat & alcohol.\n\n"
                    "Exercise & Yoga Guidelines:\n"
                    "- Gentle Movement: Chair Yoga & Light Stretching to maintain circulation.\n"
                    "- Yoga for Pain & Relaxation: Yoga Nidra (Guided Sleep Meditation).\n"
                    "- Breathing Techniques: Diaphragmatic breathing to improve oxygen intake.\n"
                )
            elif radius_mean > 15 or texture_mean > 18:
                return "Stage 3 (Advanced)", (
                    "Stage 3 (Advanced) Recommendations\n\n"
                    "Objective: Counteract side effects of aggressive treatments, minimize weight loss, and enhance tissue repair.\n\n"
                    "Dietary Recommendations:\n"
                    "- Easy-to-Digest, Nutrient-Dense Foods: Steamed vegetables, pureed soups, smoothies, oatmeal.\n"
                    "- High-Calorie, High-Protein Diet: Lean proteins (eggs, tofu, fish, chicken), nut butters, protein shakes.\n"
                    "- Anti-Nausea & Digestive Support: Ginger tea, peppermint, probiotic yogurt.\n"
                    "- Bone Health Optimization: Calcium & Vitamin D from fortified dairy, almonds, dark leafy greens.\n"
                    "- Foods to Avoid: Fried & greasy foods, excess dairy.\n\n"
                    "Exercise & Yoga Guidelines:\n"
                    "- Low-Impact Exercises: Short walks (10-15 minutes) to maintain mobility.\n"
                    "- Yoga & Breathing Techniques: Restorative Yoga with props for support.\n"
                    "- Deep Breathing: Alternate Nostril Breathing for relaxation.\n"
                )
            elif radius_mean > 12 or texture_mean > 15:
                return "Stage 2 (Moderate)", (
                    "Stage 2 (Moderate) Recommendations\n\n"
                    "Objective: Maintain strength, support chemotherapy/radiotherapy tolerance, and optimize recovery.\n\n"
                    "Dietary Recommendations:\n"
                    "- Anti-Inflammatory Foods: Turmeric, ginger, garlic, and green tea.\n"
                    "- Immune-Boosting Nutrients: Vitamin C & Zinc from citrus fruits, bell peppers, and pumpkin seeds.\n"
                    "- Protein-Rich Diet: Eggs, fish, Greek yogurt, lean meats, lentils, quinoa, tofu.\n"
                    "- Iron & Folate Sources: Spinach, legumes, lean red meat, fortified cereals.\n"
                    "- Hydration Strategies: Coconut water, bone broth, and electrolyte-rich drinks.\n"
                    "- Foods to Avoid: Spicy, greasy foods, excess caffeine.\n\n"
                    "Exercise & Yoga Guidelines:\n"
                    "- Aerobic Exercise: 30-minute moderate-intensity activities at least 5 days a week.\n"
                    "- Strength Training: Light resistance bands and bodyweight exercises.\n"
                    "- Yoga & Meditation: Yin Yoga for relaxation and lymphatic circulation.\n"
                )
            else:
                return "Stage 1 (Early)", (
                    "Stage 1 (Early) Recommendations\n\n"
                    "Objective: Strengthen immunity, reduce inflammation, and support overall well-being.\n\n"
                    "Dietary Recommendations:\n"
                    "- Nutrient-Dense Fruits & Vegetables: Broccoli, cauliflower, Brussels sprouts, carrots, sweet potatoes.\n"
                    "- Berries for Antioxidants: Blueberries, raspberries, blackberries.\n"
                    "- Whole Grains: Quinoa, brown rice, oats, whole wheat bread.\n"
                    "- Lean Protein Sources: Lentils, chickpeas, tofu, skinless poultry, wild-caught salmon.\n"
                    "- Healthy Fats: Avocados, nuts, olive oil, chia seeds, flaxseeds, walnuts.\n"
                    "- Dairy & Bone Health: Low-fat dairy or fortified plant-based alternatives.\n"
                    "- Hydration & Detox Support: 2.5-3 liters of water daily, green tea, chamomile, turmeric tea.\n"
                    "- Foods to Avoid: Processed meats, refined sugars, and excessive alcohol.\n\n"
                    "Exercise & Yoga Guidelines:\n"
                    "- Aerobic Exercise: 150 minutes per week of moderate exercise.\n"
                    "- Strength Training: 2-3 sessions per week with light resistance exercises.\n"
                    "- Yoga & Mindfulness: Hatha Yoga for flexibility and stress reduction.\n"
                    "- Pranayama (Breathwork): Improves lung function and relaxation.\n"
                )



        # Integrate Predictions in Main Code
        prediction = model.predict(input_array)
        result = " Malignant (Cancerous)" if prediction[0] == 1 else " Benign (Non-Cancerous)"
        st.subheader(f" Prediction: {result}")

        if prediction[0] == 1:
            # Predict Cancer Stage
            predicted_stage, stage_recommendation = predict_cancer_stage(input_features[0], input_features[1])  # radius_mean & texture_mean
            st.markdown(f"  Predicted Cancer Stage: {predicted_stage}")
            st.markdown(stage_recommendation)
        else:
            st.success("✅ The tumor is predicted to be Benign. However, always confirm with a medical professional.")
            st.markdown("""
            ✅ Benign Tumor Recommendations
            - Balanced Diet: Maintain a healthy lifestyle with fruits, vegetables, and lean proteins.   
            - Stay Active: Regular exercise can improve overall health.   
            - Routine Check-ups: Even benign tumors should be monitored for changes. 
            """)

        probabilities = model.predict_proba(input_array)

        benign_prob = probabilities[0][0]
        malignant_prob = probabilities[0][1]
        prediction_label = "Benign" if prediction == 0 else "Malignant"


        # ✅ Define Feature Groups
        feature_groups = ["mean", "se", "worst"]
        categories = [
            "Radius", "Texture", "Perimeter", "Area", "Smoothness",
            "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal_Dimension"
        ]
        
        input_feature_dict = dict(zip(feature_names, input_features))


        # ✅ Generate Feature Names for All 30 Parameters
        all_features = [f"{cat.lower()}_{group}" for group in feature_groups for cat in categories]
        

        # ✅ Extract Feature Values in Correct Order
        def get_feature_values(feature_list):
            return np.array([input_feature_dict.get(f, 0) for f in feature_list])

        all_values = get_feature_values(all_features).reshape(3, 10)  # Shape (3,10)
        
        # Load dataset min-max values
        csv_path = "cleaned_breast_cancer_data.csv"
        min_max_values = get_min_max_values(csv_path)
        
        # Scale the input values
        scaled_input_features = min_max_scale(input_features, min_max_values, all_features)

        scaled_input_features_dict = dict(zip(all_features, scaled_input_features.values()))

        # ✅ Extract Scaled Feature Values in Correct Order
        def get_feature_values(feature_list):
            return np.array([scaled_input_features_dict.get(f) for f in feature_list])

        all_values = get_feature_values(all_features).reshape(3, 10)  # Shape (3,10)
    

        # ✅ Normalize All 30 Features Based on Dataset Values
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(all_values.T).T  # Scale and reshape back

        # ✅ Extract Scaled Values for Mean, SE, Worst
        mean_scaled, se_scaled, worst_scaled = normalized_values

        # ✅ Create Radar Chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=mean_scaled,
            theta=categories,
            fill='toself',
            name='Mean Value',
            line_color='blue',
            opacity=0.5
        ))

        fig.add_trace(go.Scatterpolar(
            r=se_scaled,
            theta=categories,
            fill='toself',
            name='Standard Error',
            line_color='purple',
            opacity=0.5
        ))

        fig.add_trace(go.Scatterpolar(
            r=worst_scaled,
            theta=categories,
            fill='toself',
            name='Worst Value',
            line_color='red',
            opacity=0.5
        ))

        # ✅ Update Layout
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="🔍 Breast Cancer Feature Radar Chart",
            template="plotly_dark"
        )
    
        # ✅ Display in Streamlit
        st.markdown("### 📊 Breast Cancer Feature Radar Chart")
        st.plotly_chart(fig)

    
    
        # Display feature importance graph
        st.markdown("### 🔥 Feature Importance")
        feature_importance = model.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=feature_importance, y=feature_names, ax=ax, palette="coolwarm")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance in Prediction")
        st.pyplot(fig)

        # Display feature distribution graph
        st.markdown("### 📊 Feature Value Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(input_features, bins=10, kde=True, color="purple", alpha=0.7)
        ax.set_xlabel("Feature Values")
        ax.set_title("Distribution of Input Features")
        st.pyplot(fig)

        def save_feature_importance_graph():
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=feature_importance, y=feature_names, ax=ax, palette="coolwarm")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance in Prediction")
            plt.tight_layout()
            plt.savefig("feature_importance.png")

        def save_feature_distribution_graph():
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(input_features, bins=10, kde=True, color="purple", alpha=0.7)
            ax.set_xlabel("Feature Values")
            ax.set_title("Distribution of Input Features")
            plt.tight_layout()
            plt.savefig("feature_distribution.png")

        def save_radar_chart():
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=mean_scaled, theta=categories, fill='toself', name='Mean Value', line_color='blue', opacity=0.5))
            fig.add_trace(go.Scatterpolar(r=se_scaled, theta=categories, fill='toself', name='Standard Error', line_color='purple', opacity=0.5))
            fig.add_trace(go.Scatterpolar(r=worst_scaled, theta=categories, fill='toself', name='Worst Value', line_color='red', opacity=0.5))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="🔍 Breast Cancer Feature Radar Chart")
            fig.write_image("radar_chart.png")

        # Save graphs before generating the report
        save_feature_importance_graph()
        save_feature_distribution_graph()
        save_radar_chart()

        # Report Generation Feature
        def generate_report():
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, "Breast Cancer Prediction Report", ln=True, align='C')
            pdf.ln(10)

            # Patient Details Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Patient Details:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Name: {patient_name}", ln=True)
            pdf.cell(0, 8, f"Age: {patient_age}", ln=True)
            pdf.cell(0, 8, f"Gender: {patient_gender}", ln=True)  # Adding Gender
            pdf.cell(0, 8, f"Date of Examination: {date_of_examination}", ln=True)
            pdf.ln(5)
            
            # Prediction Result Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Prediction Results:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Prediction: {result}", ln=True)
            pdf.cell(0, 8, f"Cancer Stage: {predicted_stage}", ln=True)
            pdf.ln(5)
            
            # Probability Table
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Prediction Probabilities:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Probability of Benign: {benign_prob:.2%}", ln=True)
            pdf.cell(0, 8, f"Probability of Malignant: {malignant_prob:.2%}", ln=True)
            pdf.ln(5)
            
            # Recommendation Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Recommendations:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, stage_recommendation)
            pdf.ln(5)
            
            # Input Features Table Header
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Input Feature Values:", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(95, 8, "Feature Name", border=1)
            pdf.cell(95, 8, "Value", border=1, ln=True)
            
            # Input Features Table Data
            pdf.set_font("Arial", '', 12)
            for feature, value in zip(feature_names, input_features):
                pdf.cell(95, 8, feature, border=1)
                pdf.cell(95, 8, str(value), border=1, ln=True)

            # Add graphs to the PDF
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Feature Radar Chart", ln=True)
            pdf.image("radar_chart.png", x=10, w=180)
            pdf.ln(10)
            
            pdf.add_page()
            pdf.cell(200, 10, "Feature Importance", ln=True)
            pdf.image("feature_importance.png", x=10, w=180)
            pdf.ln(10)
            
            pdf.add_page()
            pdf.cell(200, 10, "Feature Value Distribution", ln=True)
            pdf.image("feature_distribution.png", x=10, w=180)
            pdf.ln(10)
            
            # Save and return PDF file
            pdf_file = "Breast_Cancer_Prediction_Report.pdf"
            pdf.output(pdf_file)
            return pdf_file


        # Download Button for Report
    pdf_file = generate_report()
    with open(pdf_file, "rb") as file:
        st.download_button(label="📥 Download Report", data=file, file_name="Breast_Cancer_Prediction_Report.pdf", mime="application/pdf")


    with col2:
        st.markdown("### Prediction Results")

        # Define button color based on prediction
        button_color = "#008000" if prediction_label == "Benign" else "#FF0000" 
        text_color = "white"  # Keep text black for better contrast

        st.markdown(
            f"""
            <style>
                .result-box {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 10px;
                    width: 100%;
                    height: auto;
                    text-align: center;
                    font-size: 18px;
                }}
                .prediction-btn {{
                    background-color: {button_color};
                    color: {text_color};
                    border: none;
                    padding: 12px 25px;
                    border-radius: 5px;
                    font-size: 18px;
                    font-weight: normal;
                    cursor: default;
                    display: inline-block;
                    margin-bottom: 10px;
                }}
            </style>
            <div class="result-box">
                <button class="prediction-btn">{prediction_label}</button>
                <p><b>Probability of Benign:</b> {benign_prob:.2%}</p>
                <p><b>Probability of Malignant:</b> {malignant_prob:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )








def main_page():
    params = st.query_params  # Get query parameters
    
    if st.session_state.get("verified") or params.get("verified", [""])[0] == "true":
        st.title("Welcome to the Lab Dashboard!")
        st.write("✅ You have successfully logged in.")
        
        # Only run the main app after verification
        run_cancer_prediction()
    else:
        verify_license()

# Call the main page function
main_page()
























