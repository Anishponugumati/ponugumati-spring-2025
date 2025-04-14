import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====================
# 1. LOAD & PREPARE DATA
# ====================

# Load your dataset CSV (update the path accordingly)
df = pd.read_csv("Preprocessed_Land_Use_Dataset.csv")

# List all columns in the dataset (for clarity)
print("Dataset columns:", df.columns.tolist())

# Define target and feature columns:
# We'll assume 'target' is the target variable
target_col = "target"
# All other columns are features (you can filter out columns like 'centroid_lat' and 'centroid_lon' if they are not desired,
# but here we include them as they might contribute spatially)
feature_cols = [col for col in df.columns if col != target_col]

# Check if any missing values exist and drop them (or impute if preferred)
df = df.dropna(subset=feature_cols + [target_col])

# Separate features and target
X = df[feature_cols]
y = df[target_col]

# ====================
# 2. PREPROCESS FEATURES
# ====================

# If you haven't already normalized your features, we scale them:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target to integer type if needed
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create PyTorch dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ====================
# 3. DEFINE THE NEURAL NETWORK
# ====================

class LandUseNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LandUseNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))  # assuming target is 0,1,...,N-1

model = LandUseNN(input_dim, num_classes)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ====================
# 4. TRAIN THE MODEL
# ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ====================
# 5. EVALUATE THE MODEL
# ====================

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
