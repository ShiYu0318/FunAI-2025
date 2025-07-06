import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# const variables

FEATURES_NUM = 4
DATA_PATH = "dataset/training_data.pkl"

# chekc if the data file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

X = [row[:FEATURES_NUM] for row in data]  # Features
y = [row[FEATURES_NUM] for row in data]   # Labels

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into train/test without scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Parameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5],
    'weights': ['uniform']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

y_pred = best_knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test set accuracy with best model: {acc:.2f}")

# Save the best model and the encoder
with open("dataset/knn_model.pkl", "wb") as f:
    pickle.dump(best_knn, f)

with open("dataset/knn_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Best model and encoder saved successfully.")
