import pandas as pd

# Load the CSV file into a DataFrame
file_path = './heart.csv'
df = pd.read_csv(file_path)

# Separate data based on 'target' values
heart_disease_data = df[df['target'] == 1]
no_heart_disease_data = df[df['target'] == 0]

# Save the separated data to CSV files
heart_disease_data.to_csv('heart_disease_data.csv', index=False)
no_heart_disease_data.to_csv('no_heart_disease_data.csv', index=False)

print("CSV files created successfully: 'heart_disease_data.csv' and 'no_heart_disease_data.csv'")
