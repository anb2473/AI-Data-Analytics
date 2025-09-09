import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
import os
import re

# --------------------------
# Load JSON dataset
# --------------------------
with open("./dataset.json") as f:
    raw_data = json.load(f)

# Convert JSON objects to strings (T5 works with sequences of text)
for entry in raw_data:
    entry["json_str"] = json.dumps(entry["json"])

# Augment the dataset by duplicating examples to help with learning
augmented_data = raw_data.copy()
# Add multiple copies to help with learning
for _ in range(10):  # Create 10 copies of each example
    for entry in raw_data:
        augmented_data.append(entry)
raw_data = augmented_data

print(f"Training with {len(raw_data)} examples")

# --------------------------
# Tokenizer & Model
# --------------------------
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --------------------------
# Dataset class
# --------------------------
class JSONDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=128, max_output_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Very explicit instruction for JSON generation
        input_text = f"Text to JSON: {item['text']}"
        target_text = item["json_str"]

        # Tokenize input and output
        input_enc = tokenizer(
            input_text, 
            max_length=self.max_input_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        target_enc = tokenizer(
            target_text, 
            max_length=self.max_output_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }

# Create dataset
dataset = JSONDataset(raw_data, tokenizer)

# --------------------------
# Custom Training Function
# --------------------------
def train_model(model, dataset, tokenizer, num_epochs=15, learning_rate=3e-4, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return model

# --------------------------
# Training
# --------------------------
print("Starting training...")
trained_model = train_model(model, dataset, tokenizer)
print("Training completed!")

# Save the model
os.makedirs("./t5-json-finetuned", exist_ok=True)
trained_model.save_pretrained("./t5-json-finetuned")
tokenizer.save_pretrained("./t5-json-finetuned")
print("Model saved to ./t5-json-finetuned/")

# --------------------------
# Inference: Generate JSON from text
# --------------------------
def generate_json(text, model, tokenizer):
    device = next(model.parameters()).device
    model.eval()
    
    input_text = f"Text to JSON: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=256,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            temperature=0.1
        )
    
    json_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Try to extract JSON from the output
    # Look for JSON-like patterns in the output
    json_match = re.search(r'\{.*\}', json_str)
    if json_match:
        json_str = json_match.group(0)
    else:
        # If no braces found, try to wrap the content
        if '"operation"' in json_str:
            json_str = "{" + json_str + "}"
    
    # Try to fix common issues
    if '"op": ""' in json_str:
        # Try to infer the operator from the text
        if "older than" in text or "greater than" in text:
            json_str = json_str.replace('"op": ""', '"op": ">"')
        elif "cheaper than" in text or "less than" in text:
            json_str = json_str.replace('"op": ""', '"op": "<"')
        elif "equal to" in text or "equals" in text:
            json_str = json_str.replace('"op": ""', '"op": "="')
    
    # Fix malformed conditions array
    if '"conditions": [' in json_str and '"field"' in json_str:
        # Convert array format to object format
        json_str = re.sub(r'"conditions": \[([^\]]+)\]', r'"conditions": [{\1}]', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON generated", "raw_output": json_str}

# --------------------------
# Example usage
# --------------------------
print("\nTesting the model...")
sample_text = "find users older than 30"
generated_json = generate_json(sample_text, trained_model, tokenizer)
print(f"Input: {sample_text}")
print(f"Generated JSON: {generated_json}")

# Test with another example
sample_text2 = "get all products cheaper than 50"
generated_json2 = generate_json(sample_text2, trained_model, tokenizer)
print(f"\nInput: {sample_text2}")
print(f"Generated JSON: {generated_json2}")

# Test with a new example to see generalization
sample_text3 = "find products with price greater than 100"
generated_json3 = generate_json(sample_text3, trained_model, tokenizer)
print(f"\nInput: {sample_text3}")
print(f"Generated JSON: {generated_json3}")