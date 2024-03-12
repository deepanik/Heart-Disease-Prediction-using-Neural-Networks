from tkinter import Tk, Label, Entry, Button, StringVar
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = keras.models.load_model('heart_disease_model.h5')

# Load the scaler used during training
scaler = joblib.load('scaler.joblib')


class HeartDiseaseGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Heart Disease Prediction")

        # Define input labels and variables
        self.feature_labels = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'Fbs',
                               'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal']
        self.feature_vars = [StringVar()
                             for _ in range(len(self.feature_labels))]
        self.feature_entries = [Entry(master, textvariable=var)
                                for var in self.feature_vars]

        # Button for prediction
        self.predict_button = Button(
            master, text="Predict", command=self.predict, bg="red")

        # Label for displaying the result
        self.result_label = Label(master, text="Prediction Result:")

        # Pack input components
        for label, entry in zip(self.feature_labels, self.feature_entries):
            label_widget = Label(master, text=label)
            label_widget.grid(sticky="w")
            entry.grid(row=self.feature_labels.index(label), column=1, pady=5)

        # Add a separator for better organization
        Label(master).grid(row=len(self.feature_labels), column=0)  # Empty row
        Label(master).grid(row=len(self.feature_labels) + 2, column=0)  # Empty row

        # Pack prediction button and result label
        self.predict_button.grid(
            row=len(self.feature_labels) + 1, column=0, columnspan=2, pady=10)
        self.result_label.grid(
            row=len(self.feature_labels) + 3, column=0, columnspan=2, pady=10)

    def predict(self):
        try:
            # Get the input data from the entry widgets
            input_data = [float(var.get()) for var in self.feature_vars]

            # Standardize the input data using the same scaler used during training
            input_data_scaled = scaler.transform([input_data])

            # Perform the prediction
            prediction = model.predict(input_data_scaled)

            # Display or use the prediction as needed
            self.result_label.config(
                text=f"Prediction Result: {prediction[0][0]:.4f}", fg='#4CAF50')
        except ValueError:
            self.result_label.config(
                text="Invalid input. Please enter numeric values for all features.", fg='#FF0000')


if __name__ == "__main__":
    # Create Tkinter window and pass it to the GUI class
    root = Tk()
    app = HeartDiseaseGUI(root)
    root.mainloop()
