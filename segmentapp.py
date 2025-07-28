import streamlit as st
import numpy as np
import joblib
import pandas as pd 

kmeans_model = joblib.load("models/Kmeans_model.pkl")
kmeans_scaler = joblib.load("models/scaler.pkl")



st.title("Customer Segmentation Input Form")

age = st.number_input("Age", min_value=20, max_value=80, value=35)

income = st.number_input("Income", min_value=0, max_value=2_000_000, value=5000)

total_spending = st.number_input(
    "Total spending or sum of purchases", min_value=0, max_value=20000, value=5000
)

num_web_purchases = st.number_input(
    "Total number of web purchases", min_value=0, max_value=100, value=10
)

num_catalog_purchases = st.number_input(
    "Total number of catalog purchases", min_value=0, max_value=100, value=5
)

num_store_purchases = st.number_input(
    "Total number of store purchases", min_value=0, max_value=100, value=5
)

numweb_visits_month = st.number_input(
    "Total number of web visits per month", min_value=0, max_value=100, value=5
)

recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=30)




input_data = pd.DataFrame({
    "Age":[age],
    "Income":[income],
    'Total_Spending':[total_spending],
    'NumWebPurchases':[num_web_purchases],
    'NumCatalogPurchases':[num_catalog_purchases],
    'NumStorePurchases':[num_store_purchases],
    "NumWebVisitsMonth":[numweb_visits_month],
    'Recency':[recency]

})

input_scaled = kmeans_scaler.transform(input_data)

# Optionally add a submit button
if st.button("Predict Segment"):
    cluster = kmeans_model.predict(input_scaled)[0]
    segment_labels = {
        0: "Budget-Conscious Casuals",
        1: "Web-Savvy Engaged Shoppers",
        2: "High-Income Catalog Buyers",
        3: "Dormant Low-Value Customers",
        4: "Older, Traditional Buyers",
        5: "Top Multi-Channel Spenders"
    }
    st.success(f"""
    Input Summary:
    - Age: {age}
    - Income: ₹{income}
    - Total Spending: ₹{total_spending}
    - Web Purchases: {num_web_purchases}
    - Catalog Purchases: {num_catalog_purchases}
    - Store Purchases: {num_store_purchases}
    - Web Visits/Month: {numweb_visits_month}
    - Recency: {recency} days
    """)
    st.markdown(f"📊 Predicted Segment: **Cluster {cluster}** — {segment_labels.get(cluster, 'Unknown')}")

    st.info("Use this segment label for personalized marketing strategies or customer insights.")

    
