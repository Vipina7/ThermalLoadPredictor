# Thermal Load Predictor

This is a **machine learning-powered web application** that predicts the thermal load (heating and cooling loads) of buildings based on various structural parameters. The app utilizes a trained model to analyze input data and estimate the building's heating and cooling load.

## 🚀 Features

- **User-friendly Interface:** Built using Streamlit for intuitive data input and visualization.
- **Machine Learning Model:** Uses a trained regression model for accurate thermal load prediction.
- **Real-Time Predictions:** Provides instant feedback based on user input.
- **Preprocessing Pipeline:** Ensures categorical and numerical inputs are properly processed before making predictions.

## 📌 How It Works

1. Users input building details like **surface area, wall area, overall height, glazing area, etc.**
2. The data is preprocessed and fed into a trained regression model.
3. The model predicts both **heating load** and **cooling load** and displays the results.

## 🧐 Installation

To run the app locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Vipina7/ThermalLoadForecasting.git

# Create a virtual environment (optional but recommended)
python -m venv (version used 3.9)
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 🏰 Input Parameters

| Parameter              | Description                                                    |
|------------------------|----------------------------------------------------------------|
| Surface Area          | Total external surface area of the building (m²)           |
| Wall Area             | Total wall area of the building (m²)                         |
| Overall Height        | Building height from ground to roof (m)                     |
| Glazing Area          | Total window area (m²)                                     |
| Glazing Area Distribution | Proportion of glazing area in different orientations |

## 📊 Dataset Description

The dataset used for training contains the following attributes:

- **Numerical Features:** Surface Area, Wall Area, Roof Area, Overall Height, Glazing Area
- **Categorical Features:** Glazing Area Distribution
- **Target Variables:**
  - **Heating Load (kWh/m²):** Energy required to maintain indoor temperature in cold weather.
  - **Cooling Load (kWh/m²):** Energy required to maintain indoor temperature in hot weather.

## 🔍 Model and Preprocessing

- **Preprocessor:**
  - Handled missing values and outliers.
  - Standardized numerical features using **Standardscaler**.
- **Model Training:**
  - Evaluated multiple models including **Linear Regression, Random Forest, XGBoost, GradientBoost etc**.
  - Selected the best-performing model based on **R-squared, RMSE, and MAE**.
  - **Final Model:** GradientBoost due to its high accuracy and generalization ability.

## 🌟 Example Output

- **Heating Load Prediction:** "The estimated heating load for this building is **24.5 kWh/m²**."
- **Cooling Load Prediction:** "The estimated cooling load for this building is **18.2 kWh/m²**."

## 📎 Folder Structure

```
ThermalLoadPredictor/
|---artifacts/
|   |--model.pkl
|   |--preprocessor.pkl
|   |--train.csv
|   |--test.csv
|   |--model_performance.csv
|___ Notebook/
|    |--Data/
|      |--thermal-load-data.csv
|    |--EDA_and_Model_Training.py
│── src/
|   |---components/
|   |    |---data_ingestion.py
|   |    |---data_transformation.py
|   |    |---model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   ├── exception.py
|   |__ logger.py
|   |__ utils.py
│── app.py
│── requirements.txt
│── README.md
```

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project!

---

🔥 **Vipina Manjunatha** 🔥
Mail me at vipina1394@gmail.com