Heart Disease Prediction

This repository contains a machine learning model to predict the likelihood of heart disease in individuals based on various health parameters. The goal of this project is to demonstrate how machine learning can be used in healthcare to assist with early diagnosis and prediction of heart disease risk.

Project Overview
Heart disease is one of the leading causes of death worldwide, and early detection can help in better management and treatment. This project uses a dataset of health-related features to train a machine learning model that predicts whether a person is at risk of heart disease or not.

Dataset
The dataset used in this project comes from the Cleveland Heart Disease Dataset, which includes various health features such as age, sex, cholesterol levels, blood pressure, and others. The dataset is available on UCI Machine Learning Repository.

Features
The model uses the following features to predict the presence of heart disease:

Age: Age of the patient

Sex: Gender of the patient (1 = male, 0 = female)

Chest pain type: Type of chest pain (categorical feature)

Resting blood pressure: Blood pressure measurement

Serum cholesterol: Cholesterol levels

Fasting blood sugar: Blood sugar level (1 = True, 0 = False)

Resting electrocardiographic results: Electrocardiographic results (categorical feature)

Maximum heart rate achieved: Maximum heart rate during exercise

Exercise induced angina: Whether exercise induced angina is present (1 = Yes, 0 = No)

ST depression induced by exercise: Depression of the ST segment

Slope of peak exercise ST segment: Slope of the exercise-induced ST segment

Number of major vessels colored by fluoroscopy: Number of vessels (0-3)

Thalassemia: Thalassemia (categorical feature)

Goal
The goal is to train a classifier to predict if a person has heart disease based on the given features. The model outputs a binary prediction: 1 (heart disease) or 0 (no heart disease).

Technologies Used
Python

Scikit-learn

Pandas

NumPy

Matplotlib

Jupyter Notebooks
