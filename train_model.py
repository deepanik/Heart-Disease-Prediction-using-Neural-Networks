import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('./heart.csv')

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for later use during prediction
joblib.dump(scaler, 'scaler.joblib')

# Build and train your neural network model (replace this with your actual model architecture and training code)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10000,
          batch_size=32, validation_split=0.2)

# Save the trained model
model.save('heart_disease_model.h5')

print("Training completed. Model and scaler saved successfully.")
