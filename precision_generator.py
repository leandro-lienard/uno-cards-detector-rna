import numpy as np

# Matriz de confusión proporcionada
confusion_matrix = np.array([
    [3, 0, 0, 2, 0, 0, 0],
    [0, 4, 0, 0, 0, 1, 0],
    [0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 1, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 5]
])

# Exactitud
total_samples = np.sum(confusion_matrix)
accuracy = np.sum(np.diag(confusion_matrix)) / total_samples

# Precisión y recall por clase
precision_recall = []
for i in range(len(confusion_matrix)):
    tp = confusion_matrix[i, i]
    fp = np.sum(confusion_matrix[:, i]) - tp
    fn = np.sum(confusion_matrix[i, :]) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision_recall.append((precision, recall))

# Resultados
print("Exactitud General:", accuracy)
print("Precisión y Recall por Clase:")
for i, (precision, recall) in enumerate(precision_recall):
    print(f"Clase {i+1}:")
    print("  Precisión:", precision)
    print("  Recall:", recall)