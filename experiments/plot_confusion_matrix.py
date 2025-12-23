import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cm = np.array([[855, 145],
               [147, 853]])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
