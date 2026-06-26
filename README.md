# 🏠 London House Price Predictor

An interactive machine learning dashboard that predicts residential property valuations across Greater London based on structural specifications and geographic location metrics.

🔗 **[Live Interactive Dashboard Link](https://house-prices-london-dwvxc8vjzqsleqkatu5teq.streamlit.app/)**

## 🚀 Project Overview
This project applies an end-to-end data science workflow to analyze and predict property values. Using historical London housing data, I engineered custom geospatial indicators and trained a Random Forest regressor to bypass traditional linear regression limitations (such as multicollinearity between room counts and floor sizes).

## 🛠️ Tech Stack
* **Language:** Python (Anaconda Environment)
* **Core Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
* **Data Visualization:** Seaborn, Matplotlib
* **Deployment:** Streamlit Community Cloud

## 📈 Methodology & Key Insights
* **Geospatial Feature Engineering:** Calculated physical distance in kilometers from each property to Charing Cross (Central London) using the Haversine formula.
* **Target Encoding:** Encoded structural London postcodes (`outcode`) relative to their baseline regional median averages to prevent extreme feature bloat.
* **Model Optimization:** Transitioned from a baseline Multiple Linear Regression model to a Random Forest Regressor (`max_depth=10`), successfully resolving an issue where the model incorrectly penalized the number of bedrooms due to continuous correlation with total floor area (`floorAreaSqM`).

## 📥 How to Run Locally
1. Clone the repository: `git clone https://github.com`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch dashboard: `streamlit run app.py`
