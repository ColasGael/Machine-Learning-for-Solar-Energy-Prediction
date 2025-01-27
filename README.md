# Machine-Learning-for-Solar-Energy-Prediction
by Adele Kuzmiakova, Gael Colas and Alex McKeehan, graduate students from Stanford University

This is our final project for the CS229: "Machine Learning" class in Stanford (2017). Our teachers were Pr. Andrew Ng and Pr. Dan Boneh.

A comprehensive project that applies machine learning techniques for predicting solar energy generation using historical weather and energy data. This project is designed to assist in improving the efficiency and reliability of solar energy forecasting.

---

## Features

- **Data Preprocessing:** Clean and preprocess weather and solar energy data for analysis.
- **Feature Engineering:** Extract and select relevant features for machine learning models.
- **Modeling:** Use machine learning algorithms (e.g., linear regression, decision trees, etc.) for prediction.
- **Evaluation:** Evaluate model performance using metrics like RMSE, MAE, and R².
- **Visualization:** Visualize data trends and prediction outputs.

---

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/saadabdullah-15/Machine-Learning-for-Solar-Energy-Prediction.git
   cd Machine-Learning-for-Solar-Energy-Prediction


Language: Python, Matlab, R

Goal: predict the hourly power production of a photovoltaic power station from the measurements of a set of weather features. 

This project could be decomposed in 3 parts:
  - Data Pre-processing: we processed the raw weather data files (input) from the National Oceanographic and Atmospheric Administration and the power production data files (output) from Urbana-Champaign solar farm to get meaningful numeric values on an hourly basis ;
  - Feature Selection: we run correlation analysis between the weather features and the energy output to discard useless features, we also implemented Principal Component Analysis to reduce the dimension of our dataset ;
  - Machine Learning : we compared the performances of our ML algorithms. Implemented models include Weighted Linear Regression with and without dimension reduction, Boosting Regression Trees, and artificial Neural Networks with and without vanishing temporal gradient

Our final report and poster are available at the root.
