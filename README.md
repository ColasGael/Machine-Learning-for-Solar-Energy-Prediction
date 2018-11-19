# Machine-Learning-for-Solar-Energy-Prediction
by Adele Kuzmiakova, Gael Colas and Alex McKeehan, graduate students from Stanford University

This is our final project for the CS229: "Machine Learning" class in Stanford (2017). Our teachers were Pr. Andrew Ng and Pr. Dan Boneh.

Language: Python, Matlab, R

Goal: predict the hourly power production of a photovoltaic power station from the measurements of a set of weather features. 

This project could be decomposed in 3 parts:
  - Data Pre-processing: we processed the raw weather data files (input) from the National Oceanographic and Atmospheric Administration and the power production data files (output) from Urbana-Champaign solar farm to get meaningful numeric values on an hourly basis ;
  - Feature Selection: we run correlation analysis between the weather features and the energy output to discard useless features, we also implemented Principal Component Analysis to reduce the dimension of our dataset ;
  - Machine Learning : we compared the performances of our ML algorithms. Implemented models include Weighted Linear Regression with and without dimension reduction, Boosting Regression Trees, and artificial Neural Networks with and without vanishing temporal gradient

Our final report and poster are available at the root.
