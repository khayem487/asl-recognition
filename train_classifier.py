import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_dict = pickle.load(open('data.pickle', 'rb'))

# Inspect the data
data = data_dict['data']
labels = data_dict['labels']

# Define labels_dict (replace with your actual labels)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Exemple : A, B, L pour les gestes

# Check lengths of all entries in data
lengths = [len(d) for d in data]
print("Lengths of data entries:", lengths)

# Determine the expected length (e.g., 42 for one hand)
expected_length = 42  # Adjust this based on your data

# Filter out entries with incorrect lengths
filtered_data = []
filtered_labels = []
for d, l in zip(data, labels):
    if len(d) == expected_length:
        filtered_data.append(d)
        filtered_labels.append(l)

# Convert to NumPy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_test, y_predict, average='weighted')  # Use 'weighted' for multi-class
f1 = f1_score(y_test, y_predict, average='weighted')  # Use 'weighted' for multi-class

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Print metrics
print("Model accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_dict.values(), yticklabels=labels_dict.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)