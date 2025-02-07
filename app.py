import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

st.title('Thermal Load Predictor App')

st.write(
    "This app predicts the heating and cooling loads of a building based on its structural and design parameters. "
    "Adjust the sliders below to input the building characteristics."
)

# User input fields with ranges based on dataset statistics
surface_area_m2 = st.number_input("Surface Area (m²)", min_value=514.5, max_value=808.5, value=673.75)
wall_area_m2 = st.number_input("Wall Area (m²)", min_value=245.0, max_value=416.5, value=318.5)
overall_height_m = st.slider("Overall Height (m)", min_value=3.5, max_value=7.0, value=5.25, step=0.5)
glazing_area_m2 = st.slider("Glazing Area (m²)", min_value=0.0, max_value=0.4, value=0.25, step=0.01)
glazing_area_Distribution = st.selectbox("Glazing Area Distribution", [0,1,2,3,4,5])

# inpu data
if st.button("Predict Energy Load"):
    try:
        data = CustomData(
            surface_area_m2,
            wall_area_m2,
            overall_height_m,
            glazing_area_m2, 
            glazing_area_Distribution
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        st.write(f"🔥 **Predicted Heating Load:** {prediction[0][0]:.2f} kWh/m²")
        st.write(f"❄️ **Predicted Cooling Load:** {prediction[0][1]:.2f} kWh/m²")
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")