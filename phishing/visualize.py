# %% 
# Cell 1: Imports and Setup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# The exact numbers you just got from train.py
rf_matrix = np.array([[3432, 6], [29, 2533]])
lr_matrix = np.array([[3421, 17], [98, 2464]])

# %%
# Cell 2: Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest Plot
sns.heatmap(rf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Legit', 'Predicted Phish'],
            yticklabels=['Actual Legit', 'Actual Phish'], ax=axes[0])
axes[0].set_title('RandomForest (Primary)\nAccuracy: 99.42%')

# Logistic Regression Plot
sns.heatmap(lr_matrix, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted Legit', 'Predicted Phish'],
            yticklabels=['Actual Legit', 'Actual Phish'], ax=axes[1])
axes[1].set_title('LogisticRegression (Baseline)\nAccuracy: 98.08%')

plt.tight_layout()
plt.show()

# %%
# Cell 3: Save for the Final Report (Crucial!)
plt.savefig('phishing/reports/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved to phishing/reports/confusion_matrices.png")