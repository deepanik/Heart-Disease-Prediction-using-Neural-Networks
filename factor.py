import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV files for Heart Disease and No Heart Disease
heart_disease_data = pd.read_csv('heart_disease_data.csv').head(5)
no_heart_disease_data = pd.read_csv('no_heart_disease_data.csv').head(5)

# Function to create bar charts and a line plot for a given DataFrame, factors, and a selected line_factor


def create_plots(data, factors, title, line_factor=None):
    num_factors = len(factors)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_factors + (1 if line_factor else 0), figsize=(15, 5))

    for i, factor in enumerate(factors):
        factor_counts = data[factor].value_counts().sort_index()
        sns.barplot(x=factor_counts.index, y=factor_counts.values,
                    ax=axes[i], palette='viridis')

        # Add data labels
        for x, y in zip(range(len(factor_counts)), factor_counts.values):
            axes[i].text(x, y + 0.1, f'{y}', ha='center')

        axes[i].set_xlabel(factor)
        axes[i].set_ylabel('Count')
        axes[i].set_title(factor)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add line plot if line_factor is specified
    if line_factor:
        line_data = data[line_factor].value_counts().sort_index()
        sns.lineplot(x=line_data.index, y=line_data.values,
                     ax=axes[-1], marker='o', color='red')
        axes[-1].set_xlabel(line_factor)
        axes[-1].set_ylabel('Count')
        axes[-1].set_title(line_factor)
        axes[-1].yaxis.grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(f'{title} Factors', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# List of factors to include in the bar charts
factors_to_plot = ['cp', 'slope', 'sex',
                   'fbs', 'restecg', 'exang', 'ca', 'thal']

# Factor for the line plot (you can change this to any factor in your dataset)
line_factor_to_plot = 'age'

# Factors of Heart Disease with line plot for age
create_plots(heart_disease_data, factors_to_plot,
             'Heart Disease', line_factor_to_plot)

# Factors of No Heart Disease with line plot for age
create_plots(no_heart_disease_data, factors_to_plot,
             'No Heart Disease', line_factor_to_plot)
