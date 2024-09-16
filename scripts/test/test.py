import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pytesseract
from PIL import Image
import requests
from io import BytesIO
from multiprocessing import Pool, cpu_count

# Define the entity-unit map for validation purposes
entity_unit_map = {
    "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "voltage": {"millivolt", "kilovolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", "pint",
                    "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
}

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocess text to extract numbers and units
def extract_number(text):
    """Extract the first number found in the text."""
    if not isinstance(text, str):
        return ""
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[0] if numbers else ""

def extract_units(text):
    """Extract the unit from the text."""
    if not isinstance(text, str):
        return ""
    units = re.findall(r'[a-zA-Z]+', text)
    return units[-1] if units else ""

# Define a function to download an image from a URL
def download_image(image_url):
    """Download an image from a URL."""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Define a function to extract text from an image using OCR
def extract_text_from_image(image):
    """Extract text from an image using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()  # Return the extracted text with leading/trailing whitespace removed
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Modify process_image to only perform image downloading and OCR
def process_image(row):
    """Process a single image, extract text."""
    image_url = row['image_link']
    entity_name = row['entity_name']
    index = row.name
    
    # Download the image
    image = download_image(image_url)
    if image:
        # Extract text from the image
        extracted_text = extract_text_from_image(image)
        return {'index': index, 'entity_name': entity_name, 'extracted_text': extracted_text}
    else:
        print(f"Failed to process image URL: {image_url}")
        return {'index': index, 'entity_name': entity_name, 'extracted_text': ''}

# Function to validate the predicted unit
def validate_unit(entity_name, unit):
    """Validate if the predicted unit is valid for the entity."""
    valid_units = entity_unit_map.get(entity_name, set())
    return unit if unit in valid_units else ""

# Main processing function
def main():
    # Load training data
    train_data_path = '/kaggle/input/amazin-ml-cleaned/train_clean.csv'
    print(f"Loading training data from {train_data_path}...")
    train_data = pd.read_csv(train_data_path)
    print(f"Number of rows in the training data: {len(train_data)}")
    train_data = train_data.head(50000)  # Use a subset for faster iteration (adjust as needed)
    print(f"Using {len(train_data)} rows for training.")

    # Prepare features and labels
    print("Preprocessing training data...")
    train_data['numerical_value'] = train_data['entity_value'].apply(extract_number)
    train_data['unit'] = train_data['entity_value'].apply(extract_units)
    print(f"Number of unique units found: {train_data['unit'].nunique()}")

    # Map units to numerical values
    unit_to_id = {unit: i for i, unit in enumerate(set(train_data['unit']))}
    id_to_unit = {i: unit for unit, i in unit_to_id.items()}
    print(f"Unit to ID mapping: {unit_to_id}")

    # Convert labels to numerical values
    train_data['unit_id'] = train_data['unit'].map(unit_to_id)
    print("Converted units to numerical IDs.")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_data[['entity_name', 'entity_value']],
        train_data['unit_id'],
        test_size=0.2,
        random_state=42
    )
    print(f"Training data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # Load pre-trained BERT model for unit prediction
    def load_bert_model():
        """Load a pre-trained BERT model for unit classification."""
        print("Loading BERT model and tokenizer...")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unit_to_id))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Move model to the appropriate device
        model.to(device)

        return model, tokenizer

    # Initialize BERT model and tokenizer
    bert_model, tokenizer = load_bert_model()

    # Tokenize and encode data
    def tokenize_data(texts, tokenizer):
        """Tokenize and encode texts."""
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings

    print("Tokenizing training and validation data...")
    train_texts = X_train['entity_value'].tolist()
    val_texts = X_val['entity_value'].tolist()
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    print("Tokenization complete.")

    # Define a DataLoader for the model
    def create_dataloader(encodings, labels, batch_size=16):
        """Create a DataLoader for the model."""
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels.values)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    train_loader = create_dataloader(train_encodings, y_train)
    val_loader = create_dataloader(val_encodings, y_val)

    # Train the model
    def train_model(model, dataloader, epochs=5):
        """Train the BERT model."""
        print("Training the BERT model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

    train_model(bert_model, train_loader)

    # Evaluate the model
    def evaluate_model(model, dataloader):
        """Evaluate the BERT model."""
        print("Evaluating the BERT model...")

        model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                # Get predictions
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Move predictions back to CPU for evaluation
                predictions = torch.argmax(logits, dim=1).cpu().tolist()
                preds.extend(predictions)

                # Move true labels back to CPU
                true_labels.extend(labels.cpu().tolist())

        # Get unique classes from the true_labels
        unique_labels = sorted(set(true_labels))

        # Create a filtered target_names list based on unique labels in true_labels
        filtered_target_names = [id_to_unit[i] for i in unique_labels]

        # Generate classification report
        report = classification_report(true_labels, preds, labels=unique_labels, target_names=filtered_target_names)
        print("Classification Report:\n", report)

        return report

    print("Evaluating the model...")
    report = evaluate_model(bert_model, val_loader)

    # Load test data
    test_data_path = '/kaggle/input/amazon-ml/test.csv'
    print(f"Loading test data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.head(100)  # Limit to 5000 rows for testing
    print(f"Number of rows in the test data: {len(test_data)}")

    # Process images to extract text using multiprocessing with progress bar
    def process_images(test_data):
        """Process images from URLs and extract text using multiprocessing."""
        print("Processing test data...")

        # Use multiprocessing to process images in parallel
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_image, [row for _, row in test_data.iterrows()]), total=len(test_data)))

        # Create a DataFrame for the results
        extracted_df = pd.DataFrame(results)
        return extracted_df

    extracted_df = process_images(test_data)

    # Now perform predictions in the main process with progress bar
    def predict_units(extracted_df):
        """Predict units for the extracted texts."""
        predictions = []
        for idx, row in tqdm(extracted_df.iterrows(), total=len(extracted_df), desc="Predicting units"):
            index = row['index']
            entity_name = row['entity_name']
            extracted_text = row['extracted_text']

            if extracted_text.strip() == '':
                predictions.append({'index': index, 'prediction': ''})
                continue

            # Extract numerical value
            extracted_num = extract_number(extracted_text)

            # Tokenize and encode
            encoded_input = tokenizer(extracted_text, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = bert_model(**encoded_input)
                predicted_unit_id = torch.argmax(outputs.logits, dim=1).item()
                predicted_unit = id_to_unit[predicted_unit_id]

            # Validate the predicted unit
            validated_unit = validate_unit(entity_name, predicted_unit)

            # Format the result for item_value
            item_value = f"{extracted_num} {validated_unit}"

            predictions.append({'index': index, 'prediction': item_value})

            print(f"Processed index {index}, Predicted value: {item_value}")

        return pd.DataFrame(predictions)

    # Get predictions
    output_df = predict_units(extracted_df)

    # Save the output
    output_csv_path = 'test_out.csv'
    output_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()