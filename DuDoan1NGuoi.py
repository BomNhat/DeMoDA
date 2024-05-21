from logging import root
import tkinter as tk
from tkinter import messagebox
from typing import Self
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

class WeightCategoryPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Weight Category Predictor")
        self.load_data()
        self.preprocess_data()
        self.create_widgets()

    def load_data(self):
        self.data = pd.read_csv("Dataset.csv")

    def preprocess_data(self):
        self.label_encoders = {}
        for column in self.data.select_dtypes(include=['object']).columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        self.scaler = StandardScaler()
        X = self.data.drop(columns=['NObeyesdad'])
        y = self.data['NObeyesdad']
        self.X_scaled = self.scaler.fit_transform(X)

    def predict(self):
        try:
            # Get user input
            user_input = {
                'Gender': self.gender_var.get(),
                'Age': float(self.age_entry.get()),
                'Height': float(self.height_entry.get()),
                'Weight': float(self.weight_entry.get()),
                'family_history_with_overweight': self.family_history_var.get(),
                'FAVC': self.favc_var.get(),
                'FCVC': float(self.fcvc_entry.get()),
                'NCP': float(self.ncp_entry.get()),
                'CAEC': self.caec_var.get(),
                'SMOKE': self.smoke_var.get(),
                'CH2O': float(self.ch2o_entry.get()),
                'SCC': self.scc_var.get(),
                'FAF': float(self.faf_entry.get()),
                'TUE': float(self.tue_entry.get()),
                'CALC': self.calc_var.get(),
                'MTRANS': self.mtrans_var.get()
            }

            # Preprocess user input
            for column in user_input:
                if column in self.label_encoders:
                    if user_input[column] not in self.label_encoders[column].classes_:
                        user_input[column] = self.label_encoders[column].transform(['Other'])[0]
                    else:
                        user_input[column] = self.label_encoders[column].transform([user_input[column]])[0]

            # Scale user input
            X_input = pd.DataFrame([user_input])
            X_input_scaled = self.scaler.transform(X_input)

            # Predict
            prediction = self.model.predict(X_input_scaled)[0]

            # Convert prediction to original label
            prediction_label = self.label_encoders['NObeyesdad'].inverse_transform([prediction])[0]

            self.result_label.config(text=f"The predicted weight category is: {prediction_label}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_widgets(self):
        # Gender
        gender_label = tk.Label(self.root, text="Gender:")
        gender_label.grid(row=0, column=0)
        self.gender_var = tk.StringVar()
        gender_options = ["Male", "Female"]
        gender_dropdown = tk.OptionMenu(self.root, self.gender_var, *gender_options)
        gender_dropdown.grid(row=0, column=1)

        # Age
        age_label = tk.Label(self.root, text="Age:")
        age_label.grid(row=1, column=0)
        self.age_entry = tk.Entry(self.root)
        self.age_entry.grid(row=1, column=1)

        # FAVC
        favc_label = tk.Label(self.root, text="FAVC:")
        favc_label.grid(row=2, column=0)
        self.favc_var = tk.StringVar()
        favc_checkbox = tk.Checkbutton(self.root, variable=self.favc_var, onvalue="yes", offvalue="no")
        favc_checkbox.grid(row=2, column=1)

# Height
height_label = tk.Label(root, text="Height (m):")
height_label.grid(row=2, column=0)
height_entry = tk.Entry(root)
height_entry.grid(row=2, column=1)

# Weight
weight_label = tk.Label(root, text="Weight (kg):")
weight_label.grid(row=3, column=0)
weight_entry = tk.Entry(root)
weight_entry.grid(row=3, column=1)

# Family history with overweight
family_history_label = tk.Label(root, text="Family history with overweight:")
family_history_label.grid(row=4, column=0)
family_history_var = tk.StringVar()
family_history_checkbox = tk.Checkbutton(root, variable=family_history_var, onvalue="yes", offvalue="no")
family_history_checkbox.grid(row=4, column=1)

# FCVC
fcvc_label = tk.Label(root, text="FCVC:")
fcvc_label.grid(row=5, column=0)
fcvc_entry = tk.Entry(root)
fcvc_entry.grid(row=5, column=1)

# NCP
ncp_label = tk.Label(root, text="NCP:")
ncp_label.grid(row=6, column=0)
ncp_entry = tk.Entry(root)
ncp_entry.grid(row=6, column=1)

# CAEC
caec_label = tk.Label(root, text="CAEC:")
caec_label.grid(row=7, column=0)
caec_var = tk.StringVar()
caec_options = ["Always", "Frequently", "Sometimes", "No"]
caec_dropdown = tk.OptionMenu(root, caec_var, *caec_options)
caec_dropdown.grid(row=7, column=1)

# SMOKE
smoke_label = tk.Label(root, text="SMOKE:")
smoke_label.grid(row=8, column=0)
smoke_var = tk.StringVar()
smoke_checkbox = tk.Checkbutton(root, variable=smoke_var, onvalue="yes", offvalue="no")
smoke_checkbox.grid(row=8, column=1)

# CH2O
ch2o_label = tk.Label(root, text="CH2O:")
ch2o_label.grid(row=9, column=0)
ch2o_entry = tk.Entry(root)
ch2o_entry.grid(row=9, column=1)

# SCC
scc_label = tk.Label(root, text="SCC:")
scc_label.grid(row=10, column=0)
scc_var = tk.StringVar()
scc_checkbox = tk.Checkbutton(root, variable=scc_var, onvalue="yes", offvalue="no")
scc_checkbox.grid(row=10, column=1)

# FAF
faf_label = tk.Label(root, text="FAF:")
faf_label.grid(row=11, column=0)
faf_entry = tk.Entry(root)
faf_entry.grid(row=11, column=1)

# TUE
tue_label = tk.Label(root, text="TUE:")
tue_label.grid(row=12, column=0)
tue_entry = tk.Entry(root)
tue_entry.grid(row=12, column=1)

# CALC
calc_label = tk.Label(root, text="CALC:")
calc_label.grid(row=13, column=0)
calc_var = tk.StringVar()
calc_options = ["Always", "Frequently", "Sometimes", "No"]
calc_dropdown = tk.OptionMenu(root, calc_var, *calc_options)
calc_dropdown.grid(row=13, column=1)

# MTRANS
mtrans_label = tk.Label(root, text="MTRANS:")
mtrans_label.grid(row=14, column=0)
mtrans_var = tk.StringVar()
mtrans_options = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
mtrans_dropdown = tk.OptionMenu(root, mtrans_var, *mtrans_options)
mtrans_dropdown.grid(row=14, column=1)

# Button to predict
predict_button = tk.Button(Self.root, text="Predict", command=Self.predict)
predict_button.grid(row=15, column=0, columnspan=2)

# Label to display result
Self.result_label = tk.Label(Self.root, text="")
Self.result_label.grid(row=16, column=0, columnspan=2)

# Create GUI
root = tk.Tk()
app = WeightCategoryPredictor(root)
root.mainloop()
