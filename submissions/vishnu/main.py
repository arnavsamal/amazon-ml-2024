import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import cv2
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from functools import partial
import urllib
import multiprocessing
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Utility Functions
def common_mistake(unit):
    if unit in allowed_units:
        return unit
    if unit.replace('ter', 'tre') in allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, allowed_units))
    return number, unit

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)
    
    create_placeholder_image(image_save_path)

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links), desc="Downloading images"))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links), desc="Downloading images"):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError("Filepath: {} invalid or not found.".format(filename))

def sanity_check(test_filename, output_filename):
    check_file(test_filename)
    check_file(output_filename)
    
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")
    
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")
    
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError("Output CSV file must contain 'index' and 'prediction' columns.")
    
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        logger.warning("Missing index in test file: {}".format(missing_index))
        
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        logger.warning("Extra index in test file: {}".format(extra_index))
        
    output_df.apply(lambda x: parse_string(x['prediction']), axis=1)
    logger.info("Parsing successful for file: {}".format(output_filename))

# Data Preparation
def download_dataset_images():
    train_df = pd.read_csv('/kaggle/input/amazon-ml-cleaned/train_clean.csv')
    test_df = pd.read_csv('/kaggle/input/amazon-ml/test.csv')

    os.makedirs('/kaggle/working/train', exist_ok=True)
    os.makedirs('/kaggle/working/test', exist_ok=True)

    logger.info("Downloading training images...")
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Downloading training images"):
        image_url = row['image_link']
        image_filename = f"/kaggle/working/train/{index}.jpg"
        download_images([image_url], '/kaggle/working/train')
        logger.debug(f"Downloaded training image {index}")

    logger.info("Downloading test images...")
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Downloading test images"):
        image_url = row['image_link']
        image_filename = f"/kaggle/working/test/{index}.jpg"
        download_images([image_url], '/kaggle/working/test')
        logger.debug(f"Downloaded test image {index}")

    logger.info("Data preparation complete!")

# Image Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_dataset(image_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(image_dir), desc=f"Preprocessing images in {image_dir}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(image_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            preprocessed_img = preprocess_image(input_path, target_size)
            cv2.imwrite(output_path, cv2.cvtColor((preprocessed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Label Preprocessing
def preprocess_labels(df):
    def extract_value_and_unit(entity_value):
        match = re.match(r'(\d+(?:\.\d+)?)\s*(\w+)', str(entity_value))
        if match:
            value, unit = match.groups()
            return float(value), unit.lower()
        return None, None

    def normalize_unit(unit, entity_name):
        allowed_units = entity_unit_map.get(entity_name, [])
        if unit in allowed_units:
            return unit
        return None

    logger.info("Preprocessing labels...")
    df['value'], df['unit'] = zip(*df['entity_value'].map(extract_value_and_unit))
    df['normalized_unit'] = df.apply(lambda row: normalize_unit(row['unit'], row['entity_name']), axis=1)
    df = df.dropna(subset=['normalized_unit'])
    
    return df

# Feature Extraction
def extract_cnn_features(image_path):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().numpy()

def extract_cnn_features_batch(image_dir):
    cnn_features = {}
    
    for filename in tqdm(os.listdir(image_dir), desc=f"Extracting CNN features from {image_dir}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image_id = os.path.splitext(filename)[0]
            
            features = extract_cnn_features(image_path)
            cnn_features[image_id] = features
    
    return cnn_features

# Model Definition
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc_cnn = nn.Linear(cnn_feature_dim, hidden_dim)
        
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, ocr, cnn):
        ocr_emb = self.embedding(ocr)
        ocr_out, _ = self.lstm(ocr_emb)
        ocr_out = ocr_out[:, -1, :]
        
        cnn_out = self.fc_cnn(cnn)
        
        combined = torch.cat((ocr_out, cnn_out), dim=1)
        output = self.fc_combined(combined)
        
        return output

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch['ocr'], batch['cnn'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch['label'].size(0)
            train_correct += predicted.eq(batch['label']).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                outputs = model(batch['ocr'], batch['cnn'])
                loss = criterion(outputs, batch['label'])
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch['label'].size(0)
                val_correct += predicted.eq(batch['label']).sum().item()
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Accuracy: {100.*train_correct/train_total:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100.*val_correct/val_total:.2f}%')

# Prediction Function
def make_predictions(model, test_loader, le):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            outputs = model(batch['ocr'], batch['cnn'])
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return le.inverse_transform(predictions)

# Error Analysis
def perform_error_analysis(val_data, val_predictions):
    logger.info("Performing error analysis...")
    print(classification_report(val_data['true_label'], val_predictions['predicted_label']))

    cm = confusion_matrix(val_data['true_label'], val_predictions['predicted_label'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    errors = val_data[val_data['true_label'] != val_predictions['predicted_label']]
    print("Example Errors:")
    for _, row in errors.head().iterrows():
        print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Image: {row['image_link']}")

    error_by_entity = errors['entity_name'].value_counts(normalize=True)
    plt.figure(figsize=(10, 6))
    error_by_entity.plot(kind='bar')
    plt.title('Error Distribution by Entity Type')
    plt.ylabel('Error Rate')
    plt.xlabel('Entity Type')
    plt.savefig('error_distribution.png')
    plt.close()

    print("Error analysis complete. Check 'confusion_matrix.png' and 'error_distribution.png' for visualizations.")

# Main execution
download_dataset_images()

# Image Preprocessing
preprocess_dataset('/kaggle/working/train', '/kaggle/working/preprocessed/train')
preprocess_dataset('/kaggle/working/test', '/kaggle/working/preprocessed/test')
print("Image preprocessing complete!")

# Label Preprocessing
train_df = pd.read_csv('/kaggle/input/amazon-ml-cleaned/train_clean.csv')
preprocessed_train_df = preprocess_labels(train_df)
preprocessed_train_df.to_csv('/kaggle/working/preprocessed/train_labels.csv', index=False)
print("Label preprocessing complete!")

# Feature Extraction
train_cnn_features = extract_cnn_features_batch('preprocessed/train')
test_cnn_features = extract_cnn_features_batch('preprocessed/test')
pd.DataFrame.from_dict(train_cnn_features, orient='index').to_csv('features/train_cnn_features.csv')
pd.DataFrame.from_dict(test_cnn_features, orient='index').to_csv('features/test_cnn_features.csv')
print("CNN feature extraction complete!")

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

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model training complete and saved!")

# Make predictions on test set
test_ocr = pd.read_csv('features/test_ocr_features.csv', index_col=0)
test_cnn = pd.read_csv('features/test_cnn_features.csv', index_col=0)
test_data = pd.concat([test_ocr, test_cnn], axis=1)
original_test = pd.read_csv('dataset/test.csv')

test_dataset = ProductDataset(test_data['ocr_text'].values, test_data.drop('ocr_text', axis=1).values, np.zeros(len(test_data)))
test_loader = DataLoader(test_dataset, batch_size=32)

predicted_values = make_predictions(model, test_loader, le)

# Post-process predictions
def format_prediction(value, entity_name):
    allowed_units = ALLOWED_UNITS.get(entity_name, [])
    if not allowed_units:
        return ""
    unit = allowed_units[0]
    return f"{value:.2f} {unit}"

# Generate output dataframe
output_df = pd.DataFrame({
    'index': original_test['index'],
    'prediction': [format_prediction(value, entity_name) 
                   for value, entity_name in zip(predicted_values, original_test['entity_name'])]
})

# Save the output file
output_file = 'test_out.csv'
output_df.to_csv(output_file, index=False)
print(f"Output file '{output_file}' generated.")

# Run sanity check
print("Running sanity check...")
run_sanity_check(output_file)
print("Sanity check complete.")

# Perform error analysis (assuming you have validation data and predictions)
val_data = pd.read_csv('validation_data.csv')
val_predictions = pd.read_csv('validation_predictions.csv')
perform_error_analysis(val_data, val_predictions)
