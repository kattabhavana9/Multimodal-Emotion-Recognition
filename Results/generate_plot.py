import matplotlib.pyplot as plt

models = ["Speech-only", "Text-only", "Fusion"]
accuracies = [0.8607, 0.1321, 0.5750]

plt.figure()
plt.bar(models, accuracies)
plt.xlabel("Model Type")
plt.ylabel("Test Accuracy")
plt.title("Comparison of Emotion Recognition Models")
plt.ylim(0, 1)

plt.savefig("accuracy_comparison.png")
plt.show()
