import streamlit as st
import numpy as np
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Japanese House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

with open("models/model_RF_param.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Japanese House Price Prediction")

st.write(
    """
    This application predicts the Total Transaction Value of a
    Japanese property using a trained machine learning regression model.
    """
)


# ============================================================
# PROPERTY INFORMATION
# ============================================================

st.header("Enter Property Details")


area = st.number_input(
    "Area",
    min_value=0.0,
    value=100.0
)

frontage = st.number_input(
    "Frontage",
    min_value=0.0,
    value=10.0
)

total_floor_area = st.number_input(
    "Total Floor Area",
    min_value=0.0,
    value=100.0
)

construction_year = st.number_input(
    "Construction Year",
    min_value=1800,
    max_value=2100,
    value=2000,
    step=1
)

building_coverage_ratio = st.number_input(
    "Building Coverage Ratio",
    min_value=0.0,
    value=50.0
)

floor_area_ratio = st.number_input(
    "Floor Area Ratio",
    min_value=0.0,
    value=100.0
)

quarter = st.number_input(
    "Quarter",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

year = st.number_input(
    "Transaction Year",
    min_value=2000,
    max_value=2100,
    value=2024,
    step=1
)


# ============================================================
# LOCATION INFORMATION
# ============================================================

municipality_category = st.number_input(
    "Municipality Category",
    min_value=0,
    value=0,
    step=1
)

average_time_to_station = st.number_input(
    "Average Time to Station",
    min_value=0.0,
    value=10.0
)

migration = st.number_input(
    "Migration",
    value=0.0
)

region_commercial_area = st.number_input(
    "Region Commercial Area",
    min_value=0.0,
    value=0.0
)

region_industrial_area = st.number_input(
    "Region Industrial Area",
    min_value=0.0,
    value=0.0
)

region_potential_residential_area = st.number_input(
    "Region Potential Residential Area",
    min_value=0.0,
    value=0.0
)

region_residential_area = st.number_input(
    "Region Residential Area",
    min_value=0.0,
    value=0.0
)


# ============================================================
# REGION
# ============================================================

region = st.selectbox(
    "Region",
    [
        "Chubu",
        "Chugoku",
        "Hokkaido",
        "Kansai",
        "Kanto",
        "Kyushu",
        "Shikoku",
        "Tohoku"
    ]
)


# ============================================================
# ENGINEERED FEATURES
# ============================================================

floor_area_greater_flag = st.selectbox(
    "Floor Area Greater Flag",
    [0, 1]
)

before_war_flag = st.selectbox(
    "Before War Flag",
    [0, 1]
)

frontage_greater_than_50 = st.selectbox(
    "Frontage Greater Than 50",
    [0, 1]
)

area_greater_flag = st.selectbox(
    "Area Greater Flag",
    [0, 1]
)


# ============================================================
# REGION ENCODING
# ============================================================

region_chubu = 0
region_chugoku = 0
region_hokkaido = 0
region_kansai = 0
region_kanto = 0
region_kyushu = 0
region_shikoku = 0
region_tohoku = 0


if region == "Chubu":
    region_chubu = 1

elif region == "Chugoku":
    region_chugoku = 1

elif region == "Hokkaido":
    region_hokkaido = 1

elif region == "Kansai":
    region_kansai = 1

elif region == "Kanto":
    region_kanto = 1

elif region == "Kyushu":
    region_kyushu = 1

elif region == "Shikoku":
    region_shikoku = 1

elif region == "Tohoku":
    region_tohoku = 1


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Transaction Value",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = np.array([[
        area,
        frontage,
        total_floor_area,
        construction_year,
        building_coverage_ratio,
        floor_area_ratio,
        quarter,
        year,
        municipality_category,
        average_time_to_station,
        migration,
        region_commercial_area,
        region_industrial_area,
        region_potential_residential_area,
        region_residential_area,
        region_chubu,
        region_chugoku,
        region_hokkaido,
        region_kansai,
        region_kanto,
        region_kyushu,
        region_shikoku,
        region_tohoku,
        floor_area_greater_flag,
        before_war_flag,
        frontage_greater_than_50,
        area_greater_flag
    ]])

    # Check feature count
    if input_data.shape[1] != len(feature_columns):

        st.error(
            f"Feature mismatch: model expects "
            f"{len(feature_columns)} features but received "
            f"{input_data.shape[1]}."
        )

        st.stop()

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    predicted_value = prediction[0]

    # Display result
    st.success("Prediction completed successfully!")

    st.metric(
        label="Predicted Total Transaction Value",
        value=f"{predicted_value:,.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Japanese House Price Prediction | Machine Learning Regression Project"
)