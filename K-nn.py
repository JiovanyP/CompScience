from random import seed, uniform, shuffle
from collections import Counter
from math import sqrt

# Generate synthetic dataset
def generate_data(n_samples=200, n_classes=3, seed_val=42):
    seed(seed_val)
    X = []
    y = []
    for class_idx in range(n_classes):
        center_x = uniform(-10, 10)
        center_y = uniform(-10, 10)
        for _ in range(n_samples // n_classes):
            point = (center_x + uniform(-1.5, 1.5), center_y + uniform(-1.5, 1.5))
            X.append(point)
            y.append(class_idx)
    return X, y

# Split dataset
def train_test_split(X, y, test_size=0.2, seed_val=42):
    seed(seed_val)
    data = list(zip(X, y))
    shuffle(data)
    split = int(len(data) * (1 - test_size))
    train, test = data[:split], data[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)

# k-NN Algorithm
def euclidean_distance(x1, x2):
    return sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)

def knn_predict(X_train, y_train, x_test, k=5):
    distances = [(euclidean_distance(x_test, x), label) for x, label in zip(X_train, y_train)]
    distances.sort(key=lambda d: d[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

def knn_accuracy(X_train, y_train, X_test, y_test, k=5):
    correct = sum(knn_predict(X_train, y_train, x, k) == y for x, y in zip(X_test, y_test))
    return correct / len(y_test)

# Generate data and split
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Evaluate model
accuracy = knn_accuracy(X_train, y_train, X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
