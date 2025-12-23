import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_true = np.concatenate([np.zeros(1000), np.ones(1000)])

# Approximate prediction probabilities (based on your accuracy)
y_scores = np.concatenate([
    np.random.uniform(0.6, 0.9, 1000),
    np.random.uniform(0.6, 0.9, 1000)
])

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
