import matplotlib.pyplot as plt

features = ["GGF", "FFT", "GLCM", "Fused"]
dimensions = [272, 1024, 4, 1300]

plt.figure()
plt.bar(features, dimensions)
plt.ylabel("Feature Dimension")
plt.title("Feature Dimension Comparison")
plt.show()
