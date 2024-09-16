import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ProductDataset(Dataset):
    def __init__(self, ocr_features, cnn_features, labels):
        self.ocr_features = ocr_features
        self.cnn_features = cnn_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'ocr': self.ocr_features[idx],
            'cnn': self.cnn_features[idx],
            'label': self.labels[idx]
        }

class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes):
        super(HybridModel, self).__init__()
        
        # OCR branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # CNN branch
        self.fc_cnn = nn.Linear(cnn_feature_dim, hidden_dim)
        
        # Combined layers
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, ocr, cnn):
        # OCR branch
        ocr_emb = self.embedding(ocr)
        ocr_out, _ = self.lstm(ocr_emb)
        ocr_out = ocr_out[:, -1, :]  # Take the last output
        
        # CNN branch
        cnn_out = self.fc_cnn(cnn)
        
        # Combine and predict
        combined = torch.cat((ocr_out, cnn_out), dim=1)
        output = self.fc_combined(combined)
        
        return output

# Load features and labels
train_ocr = pd.read_csv('features/train_ocr_features.csv', index_col=0)
train_cnn = pd.read_csv('features/train_cnn_features.csv', index_col=0)
train_labels = pd.read_csv('preprocessed/train_labels.csv')

# Prepare data
le = LabelEncoder()
train_labels['encoded_value'] = le.fit_transform(train_labels['value'])

# Split data
train_data, val_data, train_labels, val_labels = train_test_split(
    pd.concat([train_ocr, train_cnn], axis=1),
    train_labels['encoded_value'],
    test_size=0.2,
    random_state=42
)

# Create datasets and dataloaders
train_dataset = ProductDataset(train_data['ocr_text'].values, train_data.drop('ocr_text', axis=1).values, train_labels.values)
val_dataset = ProductDataset(val_data['ocr_text'].values, val_data.drop('ocr_text', axis=1).values, val_labels.values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
vocab_size = 10000  # Adjust based on your vocabulary
embedding_dim = 100
hidden_dim = 128
cnn_feature_dim = train_cnn.shape[1]
num_classes = len(le.classes_)

model = HybridModel(vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['ocr'], batch['cnn'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['ocr'], batch['cnn'])
            loss = criterion(outputs, batch['label'])
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch['label'].size(0)
            correct += predicted.eq(batch['label']).sum().item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {loss.item():.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Accuracy: {100.*correct/total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model training complete and saved!")

import cv2
import pytesseract
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

def perform_ocr(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Perform text extraction
    text = pytesseract.image_to_string(threshold)
    
    return text

def extract_ocr_features(image_dir):
    ocr_features = {}
    
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image_id = os.path.splitext(filename)[0]
            
            ocr_text = perform_ocr(image_path)
            ocr_features[image_id] = ocr_text
    
    return ocr_features

# Extract OCR features for training and test sets
train_ocr_features = extract_ocr_features('preprocessed/train')
test_ocr_features = extract_ocr_features('preprocessed/test')

# Save OCR features
pd.DataFrame.from_dict(train_ocr_features, orient='index', columns=['ocr_text']).to_csv('features/train_ocr_features.csv')
pd.DataFrame.from_dict(test_ocr_features, orient='index', columns=['ocr_text']).to_csv('features/test_ocr_features.csv')

print("OCR feature extraction complete!")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time

def time_inference(model, dataloader):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch['ocr'], batch['cnn'])
    end_time = time.time()
    return end_time - start_time

# Load the model
model = HybridModel(vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes)
model.load_state_dict(torch.load('model.pth'))

# Create a sample dataloader
sample_dataset = ProductDataset(...)  # Fill with sample data
sample_dataloader = DataLoader(sample_dataset, batch_size=32)

# Time the original model
original_time = time_inference(model, sample_dataloader)
print(f"Original inference time: {original_time:.4f} seconds")

# Optimize the model
optimized_model = torch.jit.script(model)

# Time the optimized model
optimized_time = time_inference(optimized_model, sample_dataloader)
print(f"Optimized inference time: {optimized_time:.4f} seconds")

# Save the optimized model
torch.jit.save(optimized_model, 'optimized_model.pth')

print("Performance optimization complete. Optimized model saved as 'optimized_model.pth'")

