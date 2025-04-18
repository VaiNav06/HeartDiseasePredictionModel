import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import urllib.request
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
import time

# Download and load dataset
url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
response = urllib.request.urlopen(url)
data = np.genfromtxt(response, delimiter=",", skip_header=1, dtype=str)

# Extract headers
with urllib.request.urlopen(url) as response:
    headers = response.readline().decode("utf-8").strip().split(",")

# Convert categorical column to numerical
categorical_column_index = headers.index("thal")
label_encoder = LabelEncoder()
data[:, categorical_column_index] = label_encoder.fit_transform(data[:, categorical_column_index])

# Separate features and target
features = data[:, :-1].astype(float)
target = data[:, -1].astype(float)

# Select top 8 features using Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=8)
features = rfe.fit_transform(features, target)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define neural network
class HeartDiseaseNN(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Initialize model, optimizer, and loss function
model = HeartDiseaseNN(input_size=X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
criterion = nn.BCELoss()

# Prepare data for training
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training parameters
epochs = 100
patience = 10  # Stop training if no improvement for 10 epochs
best_loss = float("inf")
train_losses = []
test_losses = []

start_time = time.time()

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())

    scheduler.step(test_loss)

    # Early stopping check
    if test_losses[-1] < best_loss:
        best_loss = test_losses[-1]
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    if early_stop_counter >= patience:
        print(f"Early stopping after {epoch+1} epochs.")
        break

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Plot training and test losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Losses')
plt.show()

# Evaluate final model accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).round()
    accuracy = (predictions.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)

print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
