import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="nirmalhugface/super_kart_sales_model",
    filename="best_super_kart_sales_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("SuperKart Sales Forecasting")
st.write("Enter the product and store details to predict the product store sales.")

Product_Id = st.text_input("Product ID", "FDX07")

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=10.0
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "Regular",
        "No Sugar",
        "reg"
    ]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.10,
    format="%.4f"
)

Product_Type = st.selectbox(
    "Product Type",
    [
        "Meat",
        "Snack Foods",
        "Hard Drinks",
        "Dairy",
        "Canned",
        "Soft Drinks",
        "Health and Hygiene",
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Household",
        "Seafood",
        "Starchy Foods",
        "Others"
    ]
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=100.0
)

Store_Id = st.text_input(
    "Store ID",
    "OUT049"
)

Store_Establishment_Year = st.number_input(
    "Store Establishment Year",
    min_value=1980,
    max_value=2025,
    value=2005
)

Store_Size = st.selectbox(
    "Store Size",
    [
        "High",
        "Medium",
        "Small"
    ]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Food Mart"
    ]
)

input_data = pd.DataFrame([{
    "Product_Id": Product_Id,
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}])

if st.button("Predict Sales"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Product Store Sales: {prediction:.2f}")
