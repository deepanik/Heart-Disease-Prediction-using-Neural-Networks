import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, auc, f1_score, classification_report
from tensorflow import keras
import joblib

# Load the heart disease dataset
file_path = 'heart.csv'
df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features using the pre-trained scaler
scaler = joblib.load('scaler.joblib')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the pre-trained model
model = keras.models.load_model('heart_disease_model.h5')

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
y_pred_binary = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy on the test set: {accuracy:.4f}')

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

# Plot Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='darkorange', lw=2,
         label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.savefig('precision_recall_curve.png')
plt.show()

# F1 Score
f1 = f1_score(y_test, y_pred_binary)
print(f'F1 Score: {f1:.4f}')

# Classification Report
classification_rep = classification_report(y_test, y_pred_binary)
print("Classification Report:")
print(classification_rep)

# Save Classification Report to a text file
with open('classification_report.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(classification_rep)
    print("Classification Report saved to 'classification_report.txt'")

# Line Chart
plt.figure(figsize=(10, 6))
plt.plot(df['age'], df['thalach'], 'o', alpha=0.5, label='Data Points')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.title('Age vs Maximum Heart Rate')
plt.legend()
plt.savefig('line_chart.png')
plt.show()

# Pie Chart
target_counts = df['target'].value_counts()
labels = ['No Heart Disease', 'Heart Disease']
colors = ['#66b3ff', '#ff9999']
plt.figure(figsize=(8, 8))
plt.pie(target_counts, labels=labels,
        autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of Heart Disease')
plt.savefig('pie_chart.png')
plt.show()
