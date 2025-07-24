import joblib as jb
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Return Prediction", page_icon="📦")

# Load model and scaler
try:
    model = jb.load('customer_return_model.pkl')
    scaler = jb.load('scaler.pkl')
    st.success("Model & Scaler Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("📦 Customer Return Prediction – AmazonCU")
st.markdown("Fill in the order details to predict whether the product will be returned.")

# ------------------------------
# 🔹 Input from the user
# ------------------------------
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Books", "Beauty"])
price = st.number_input("Price (in ₹)", min_value=50.0, max_value=10000.0, value=499.0)
delivery_days = st.slider("Delivery Days", 1, 10, 3)
customer_tier = st.selectbox("Customer Tier", ["Bronze", "Silver", "Gold", "Platinum"])
is_cod = st.selectbox("Cash on Delivery", ["Yes", "No"])
product_rating = st.slider("Product Rating", 1.0, 5.0, 4.0)
product_weight_grams = st.slider("Product Weight (grams)", 100, 5000, 1200)
customer_location = st.selectbox("Customer Location", ["North", "South", "East", "West", "Central"])
days_to_return = st.slider("Days to Return (if any)", 0, 30, 0)
return_reason = st.selectbox("Return Reason", ["None", "Defective", "Wrong Item", "Changed Mind", "Late Delivery", "Other"])

# ------------------------------
# 🔹 Encoding categorical inputs
# ------------------------------
category_map = {"Electronics": 0, "Clothing": 1, "Home": 2, "Books": 3, "Beauty": 4}
tier_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
location_map = {"North": 0, "South": 1, "East": 2, "West": 3, "Central": 4}
reason_map = {"None": 0, "Defective": 1, "Wrong Item": 2, "Changed Mind": 3, "Late Delivery": 4, "Other": 5}
is_cod_val = 1 if is_cod == "Yes" else 0

# ------------------------------
# 🔹 Exact feature order for prediction
# ------------------------------
column_order = [
    'product_category',
    'price',
    'delivery_days',
    'customer_tier',
    'is_cod',
    'product_rating',
    'customer_location',
    'return_reason',
    'product_weight_grams',
    'days_to_return'
]

input_values = [
    category_map[product_category],
    price,
    delivery_days,
    tier_map[customer_tier],
    is_cod_val,
    product_rating,
    location_map[customer_location],
    reason_map[return_reason],
    product_weight_grams,
    days_to_return
]

input_df = pd.DataFrame([input_values], columns=column_order)

# ------------------------------
# 🔹 Predict
# ------------------------------
if st.button("Predict Return"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = "🔁 Return" if prediction[0] == 1 else "✅ Not Return"
        st.subheader("Prediction Result")
        st.info(f"The probability of product is likely to be: **{result}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
