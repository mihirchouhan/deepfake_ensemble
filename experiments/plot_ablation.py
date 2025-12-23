import matplotlib.pyplot as plt

features = ["GGF Only", "FFT Only", "GLCM Only", "GGF + FFT + GLCM"]
accuracy = [0.71, 0.76, 0.68, 0.85]

plt.figure()
plt.bar(features, accuracy)
plt.ylabel("Accuracy")
plt.title("Ablation Study of Feature Contributions")
plt.ylim(0.6, 0.9)
plt.show()
