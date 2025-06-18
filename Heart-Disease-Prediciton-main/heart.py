import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(r"C:\Users\motir\Desktop\Project\Heart-Disease-Prediciton-main\heart.csv")


data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
data['sex'] = data['sex'].astype('category')
data['cp'] = data['cp'].astype('category')
data['fbs'] = data['fbs'].astype('category')
data['restecg'] = data['restecg'].astype('category')
data['exang'] = data['exang'].astype('category')
data['slope'] = data['slope'].astype('category')
data['thal'] = data['thal'].astype('category')

X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict():
    age = float(age_entry.get())
    sex = sex_var.get()
    cp = cp_var.get()
    trestbps = float(trestbps_entry.get())
    chol = float(chol_entry.get())
    fbs = fbs_var.get()
    restecg = restecg_var.get()
    thalach = float(thalach_entry.get())
    exang = exang_var.get()
    oldpeak = float(oldpeak_entry.get())
    slope = slope_var.get()
    ca = float(ca_entry.get())
    thal = thal_var.get()
    prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    if prediction == 0:
        result_label.config(text="Prediction: No Heart Disease")
    else:
        result_label.config(text="Prediction: Heart Disease Present")
root = tk.Tk()
root.title("Heart Disease Prediction")

age_label = ttk.Label(root, text="Age:")
age_label.grid(row=0, column=0, padx=5, pady=5)
age_entry = ttk.Entry(root)
age_entry.grid(row=0, column=1, padx=5, pady=5)

sex_label = ttk.Label(root, text="Sex (0: Female, 1: Male):")
sex_label.grid(row=1, column=0, padx=5, pady=5)
sex_var = tk.IntVar()
sex_combo = ttk.Combobox(root, textvariable=sex_var, values=[0, 1])
sex_combo.grid(row=1, column=1, padx=5, pady=5)

cp_label = ttk.Label(root, text="Chest Pain Type (0-3):")
cp_label.grid(row=2, column=0, padx=5, pady=5)
cp_var = tk.IntVar()
cp_combo = ttk.Combobox(root, textvariable=cp_var, values=[0, 1, 2, 3])
cp_combo.grid(row=2, column=1, padx=5, pady=5)

trestbps_label = ttk.Label(root, text="Resting Blood Pressure:")
trestbps_label.grid(row=3, column=0, padx=5, pady=5)
trestbps_entry = ttk.Entry(root)
trestbps_entry.grid(row=3, column=1, padx=5, pady=5)

chol_label = ttk.Label(root, text="Cholesterol:")
chol_label.grid(row=4, column=0, padx=5, pady=5)
chol_entry = ttk.Entry(root)
chol_entry.grid(row=4, column=1, padx=5, pady=5)

fbs_label = ttk.Label(root, text="Fasting Blood Sugar (0: < 120 mg/dl, 1: >= 120 mg/dl):")
fbs_label.grid(row=5, column=0, padx=5, pady=5)
fbs_var = tk.IntVar()
fbs_combo = ttk.Combobox(root, textvariable=fbs_var, values=[0, 1])
fbs_combo.grid(row=5, column=1, padx=5, pady=5)

restecg_label = ttk.Label(root, text="Resting Electrocardiographic Results (0-2):")
restecg_label.grid(row=6, column=0, padx=5, pady=5)
restecg_var = tk.IntVar()
restecg_combo = ttk.Combobox(root, textvariable=restecg_var, values=[0, 1, 2])
restecg_combo.grid(row=6, column=1, padx=5, pady=5)

thalach_label = ttk.Label(root, text="Maximum Heart Rate Achieved:")
thalach_label.grid(row=7, column=0, padx=5, pady=5)
thalach_entry = ttk.Entry(root)
thalach_entry.grid(row=7, column=1, padx=5, pady=5)

exang_label = ttk.Label(root, text="Exercise Induced Angina (0: No, 1: Yes):")
exang_label.grid(row=8, column=0, padx=5, pady=5)
exang_var = tk.IntVar()
exang_combo = ttk.Combobox(root, textvariable=exang_var, values=[0, 1])
exang_combo.grid(row=8, column=1, padx=5, pady=5)

oldpeak_label = ttk.Label(root, text="ST Depression Induced by Exercise Relative to Rest:")
oldpeak_label.grid(row=9, column=0, padx=5, pady=5)
oldpeak_entry = ttk.Entry(root)
oldpeak_entry.grid(row=9, column=1, padx=5, pady=5)

slope_label = ttk.Label(root, text="Slope of the Peak Exercise ST Segment (0-2):")
slope_label.grid(row=10, column=0, padx=5, pady=5)
slope_var = tk.IntVar()
slope_combo = ttk.Combobox(root, textvariable=slope_var, values=[0, 1, 2])
slope_combo.grid(row=10, column=1, padx=5, pady=5)

ca_label = ttk.Label(root, text="Number of Major Vessels Colored by Fluoroscopy:")
ca_label.grid(row=11, column=0, padx=5, pady=5)
ca_entry = ttk.Entry(root)
ca_entry.grid(row=11, column=1, padx=5, pady=5)

thal_label = ttk.Label(root, text="Thal (0-3):")
thal_label.grid(row=12, column=0, padx=5, pady=5)
thal_var = tk.IntVar()
thal_combo = ttk.Combobox(root, textvariable=thal_var, values=[0, 1, 2, 3])
thal_combo.grid(row=12, column=1, padx=5, pady=5)

predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=13, columnspan=2, padx=5, pady=5)

result_label = ttk.Label(root, text="")
result_label.grid(row=14, columnspan=2, padx=5, pady=5)

root.mainloop()
