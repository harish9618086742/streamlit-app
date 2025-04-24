
import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

# Load model and encoder
model = joblib.load("C:\\Users\\bashp\\Downloads\\my_project\\myenv\\fraud_detection_model.jb")
encoder = joblib.load("C:\\Users\\bashp\\Downloads\\my_project\\myenv\\label_encoders.jb")

# Distance function
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# --- Page Functions ---

def home():
    st.title("🏠 Welcome to Fraud Detection System")
    st.write("Navigate from the sidebar to:")
    st.markdown("- 🔍 *Check single transactions for fraud*")
    st.markdown("- 📤 *Upload a CSV file for batch prediction*")
    st.markdown("- 👤 *View your profile*")
    st.info("Make sure your model and encoders are loaded correctly to get started!")

def my_profile():
    st.title("👤 My Profile")
    st.write("This is a placeholder for user profile details.")
    # Add fake profile info here
    st.text("Name: REVANTH")
    st.text("Email: revanth@example.com")
    st.text("Role: Fraud Analyst")
    st.text("Member since: Jan 2024")

def fraud_check():
    st.title("💳 Check Single Transaction")
    col1, col2 = st.columns(2)
    with col1:
    
    # Autofill button
      if st.button("Autofill Example 1"):
        st.session_state.merchant = "Amazon"
        st.session_state.category = "Shopping"
        st.session_state.amt = 129.99
        st.session_state.lat = 40.7128
        st.session_state.long = -74.0060
        st.session_state.merch_lat = 40.7580
        st.session_state.merch_long = -73.9855
        st.session_state.hour = 14
        st.session_state.day = 10
        st.session_state.month = 4
        st.session_state.gender = "Female"
        st.session_state.cc_num = "1234567812345678"
    with col2:
     if st.button("Autofill Example 2"):
        st.session_state.merchant = "fraud_Rutherford-Mertz"
        st.session_state.category = "grocery_pos"
        st.session_state.amt = 281.06
        st.session_state.lat = 35.9946
        st.session_state.long = -118.2437
        st.session_state.merch_lat = 36.430124
        st.session_state.merch_long = -81.17948299999999
        st.session_state.hour = 1
        st.session_state.day = 2
        st.session_state.month = 1
        st.session_state.gender = "Male"
        st.session_state.cc_num = "4613314721966"

    # Inputs
    merchant = st.text_input("Merchant Name", key="merchant")
    category = st.text_input("Category", key="category")
    amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f", key="amt")
    lat = st.number_input("Latitude", format="%.6f", key="lat")
    long = st.number_input("Longitude", format="%.6f", key="long")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f", key="merch_lat")
    merch_long = st.number_input("Merchant Longitude", format="%.6f", key="merch_long")
    hour = st.slider("Transaction Hour", 0, 23, 12, key="hour")
    day = st.slider("Transaction Day", 1, 31, 15, key="day")
    month = st.slider("Transaction Month", 1, 12, 6, key="month")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    cc_num = st.text_input("Credit Card Number", type="password", key="cc_num")

    distance = haversine(lat, long, merch_lat, merch_long)

    if st.button("Check For Fraud"):
        if merchant and category and cc_num:
            input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                      columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

            for col in ['merchant', 'category', 'gender']:
                try:
                    input_data[col] = encoder[col].transform(input_data[col])
                except:
                    input_data[col] = -1

            input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
            prediction = model.predict(input_data)[0]
            result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
            st.success(f"Prediction: {result}")
        else:
            st.error("Please fill all required fields.")

def batch_upload():
    st.title("📤 Batch CSV Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['merchant', 'category', 'amt', 'lat', 'long', 'merch_lat', 'merch_long', 
                             'hour', 'day', 'month', 'gender', 'cc_num']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            else:
                df['distance'] = df.apply(
                    lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']),
                    axis=1
                )
                df.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1, inplace=True)

                for col in ['merchant', 'category', 'gender']:
                    try:
                        df[col] = encoder[col].transform(df[col])
                    except:
                        df[col] = -1

                df['cc_num'] = df['cc_num'].apply(lambda x: hash(str(x)) % (10 ** 2))
                df['Prediction'] = model.predict(df)
                df['Prediction'] = df['Prediction'].apply(lambda x: "Fraudulent" if x == 1 else "Legitimate")

                st.success("Batch Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Sidebar Navigation ---
st.sidebar.title("🔎 Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Fraud Check", "Batch Upload", "My Profile"])

if page == "Home":
    home()
elif page == "Fraud Check":
    fraud_check()
elif page == "Batch Upload":
    batch_upload()
elif page == "My Profile":
    my_profile()
