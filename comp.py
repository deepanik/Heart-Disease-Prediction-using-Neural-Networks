import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = './heart.csv'
df = pd.read_csv(file_path)

# Plot 1: Pie chart for distribution of heart disease
target_counts = df['target'].value_counts()
labels = ['No Heart Disease', 'Heart Disease']
colors_pie = ['#66b3ff', '#ff9999']  # Blue and red shades for better contrast

plt.figure(figsize=(12, 6))

# Subplot 1: Pie chart
plt.subplot(1, 2, 1)
plt.pie(target_counts, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors_pie, wedgeprops=dict(width=0.4),
        textprops={'fontsize': 12, 'color': 'red'})
plt.title('Distribution of Heart Disease', fontsize=16, pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Plot 2: Scatter plot for relationship between age and maximum heart rate
colors_scatter = ['#3498db', '#e74c3c']  # Lighter shades for scatter plot

# Subplot 2: Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(df[df['target'] == 0]['age'], df[df['target'] == 0]['thalach'],
            label='No Heart Disease', color=colors_scatter[0], alpha=0.7, s=80)

plt.scatter(df[df['target'] == 1]['age'], df[df['target'] == 1]['thalach'],
            label='Heart Disease', color=colors_scatter[1], alpha=0.7, s=80)

plt.xlabel('Age', fontsize=14)
plt.ylabel('Maximum Heart Rate (thalach)', fontsize=14)
plt.title('Age vs Maximum Heart Rate', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
