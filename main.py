import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}

.title-box h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 5px;
}

.title-box p {
    color: white;
    font-size: 17px;
}

.section-title {
    background-color: #667eea;
    color: white;
    padding: 10px 15px;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 15px;
    font-size: 20px;
    font-weight: bold;
}

.result-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 25px;
}

.result-box h2 {
    color: white;
    font-size: 30px;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL AND FEATURE COLUMNS
# ---------------------------------------------------

@st.cache_resource
def load_prediction_files():

    model = joblib.load("rent_model.pkl")

    feature_columns = joblib.load("feature_columns.pkl")

    return model, feature_columns


model, feature_columns = load_prediction_files()


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv(
    r"C:\Users\Administrator\Downloads\House_Rent_Dataset.csv"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<div class="title-box">
    <h1>🏠 House Rent Prediction</h1>
    <p>Enter the house details to estimate the monthly rent</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 Property Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    BHK = st.number_input(
        "BHK",
        min_value=int(df["BHK"].min()),
        max_value=int(df["BHK"].max()),
        value=2,
        step=1
    )


with col2:

    Bathroom = st.number_input(
        "Bathroom",
        min_value=int(df["Bathroom"].min()),
        max_value=int(df["Bathroom"].max()),
        value=2,
        step=1
    )


Size = st.number_input(
    "Size (sq.ft)",
    min_value=int(df["Size"].min()),
    max_value=int(df["Size"].max()),
    value=850,
    step=10
)


# ---------------------------------------------------
# LOCATION INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">📍 Location Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    City = st.selectbox(
        "City",
        sorted(
            df["City"].dropna().unique().tolist()
        )
    )


with col2:

    Area_Type = st.selectbox(
        "Area Type",
        sorted(
            df["Area Type"].dropna().unique().tolist()
        )
    )


Area_Locality = st.selectbox(
    "Area Locality",
    sorted(
        df["Area Locality"].dropna().unique().tolist()
    )
)


# ---------------------------------------------------
# FLOOR INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🏢 Floor Information</div>',
    unsafe_allow_html=True
)

Floor = st.selectbox(
    "Floor",
    sorted(
        df["Floor"].dropna().unique().tolist(),
        key=str
    )
)


# ---------------------------------------------------
# HOUSE DETAILS
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🛋️ House Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    Furnishing_Status = st.selectbox(
        "Furnishing Status",
        sorted(
            df["Furnishing Status"].dropna().unique().tolist()
        )
    )


with col2:

    Tenant_Preferred = st.selectbox(
        "Tenant Preferred",
        sorted(
            df["Tenant Preferred"].dropna().unique().tolist()
        )
    )


# ---------------------------------------------------
# POINT OF CONTACT
# ---------------------------------------------------

Point_of_Contact = st.selectbox(
    "Point of Contact",
    sorted(
        df["Point of Contact"].dropna().unique().tolist()
    )
)


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)


if st.button("🔮 Predict Rent"):

    # -----------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------

    input_data = pd.DataFrame([{

        "BHK": BHK,

        "Size": Size,

        "Floor": Floor,

        "Area Type": Area_Type,

        "Area Locality": Area_Locality,

        "City": City,

        "Furnishing Status": Furnishing_Status,

        "Tenant Preferred": Tenant_Preferred,

        "Bathroom": Bathroom,

        "Point of Contact": Point_of_Contact

    }])


    # -----------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------

    categorical_columns = [

        "Floor",

        "Area Type",

        "Area Locality",

        "City",

        "Furnishing Status",

        "Tenant Preferred",

        "Point of Contact"

    ]


    input_data = pd.get_dummies(

        input_data,

        columns=categorical_columns,

        dtype=int

    )


    # -----------------------------------------------
    # MATCH TRAINING FEATURES
    # -----------------------------------------------

    input_data = input_data.reindex(

        columns=feature_columns,

        fill_value=0

    )


    # -----------------------------------------------
    # PREDICTION
    # -----------------------------------------------

    prediction = model.predict(input_data)


    # -----------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------

    st.markdown(f"""
    <div class="result-box">
        <h2>🏠 Estimated Monthly Rent</h2>
        <h2>₹ {prediction[0]:,.0f}</h2>
        <p>Based on the house details you entered</p>
    </div>
    """, unsafe_allow_html=True)