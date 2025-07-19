#!/usr/bin/env python3
"""
Script to fine-tune the T5 query rewriting model using collected training data.
This demonstrates how to use the query rewriting training data to improve the model.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np

class QueryRewritingDataset(Dataset):
    """Dataset for T5 query rewriting training from collected data"""
    
    def __init__(self, training_data: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        
        # Process training data into input-target pairs
        for entry in training_data:
            original_query = entry["original_query"]
            selected_sources = entry["selected_sources"]
            query_source_mapping = entry["query_source_mapping"]
            
            # Find which queries led to the selected sources
            successful_queries = []
            for query, query_info in query_source_mapping.items():
                query_sources = query_info.get("sources", [])
                # Check if any of the selected sources came from this query
                for source in query_sources:
                    if source["url"] in selected_sources:
                        successful_queries.append(query)
                        break
            
            # Create training pairs: original query -> successful query
            for successful_query in successful_queries:
                if successful_query != original_query:  # Skip if same as original
                    self.inputs.append(original_query)
                    self.targets.append(successful_query)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten()
        }

def load_training_data() -> List[Dict]:
    """Load query rewriting training data from file"""
    training_file = Path("../training_data/query_rewriting_training_data.json")
    if not training_file.exists():
        print("No query rewriting training data found. Run some queries in training mode first.")
        return []
    
    with open(training_file, 'r') as f:
        data = json.load(f)
        return data.get("query_rewriting_data", [])

def prepare_training_data(training_data: List[Dict]) -> Tuple[List, List]:
    """Prepare training data for the T5 query rewriting model"""
    print(f"Processing {len(training_data)} training entries...")
    
    inputs = []
    targets = []
    
    for entry in training_data:
        original_query = entry["original_query"]
        selected_sources = entry["selected_sources"]
        query_source_mapping = entry["query_source_mapping"]
        
        print(f"Original query: {original_query}")
        print(f"  Selected sources: {len(selected_sources)}")
        
        # Find which queries led to the selected sources
        successful_queries = []
        for query, query_info in query_source_mapping.items():
            query_sources = query_info.get("sources", [])
            # Check if any of the selected sources came from this query
            for source in query_sources:
                if source["url"] in selected_sources:
                    successful_queries.append(query)
                    break
        
        print(f"  Successful queries: {len(successful_queries)}")
        
        # Create training pairs: original query -> successful query
        for successful_query in successful_queries:
            if successful_query != original_query:  # Skip if same as original
                inputs.append(original_query)
                targets.append(successful_query)
    
    print(f"Total training pairs: {len(inputs)}")
    
    return inputs, targets

def demonstrate_query_rewriting_training():
    """Demonstrate how to use the training data for T5 query rewriting fine-tuning"""
    
    # Load training data
    training_data = load_training_data()
    if not training_data:
        print("No training data available. Please run some queries in training mode first.")
        return
    
    # Prepare training data
    inputs, targets = prepare_training_data(training_data)
    
    if len(inputs) == 0:
        print("No valid training pairs found.")
        return
    
    # Load the T5 model and tokenizer
    print("Loading T5 query rewriting model...")
    model_id = "prhegde/t5-query-reformulation-RL"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    
    print("\n=== Training Data Preview ===")
    print("Sample training pairs:")
    for i in range(min(3, len(inputs))):
        print(f"  Input: {inputs[i]}")
        print(f"  Target: {targets[i]}")
        print()
    
    # Create dataset
    dataset = QueryRewritingDataset(training_data, tokenizer)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Demonstrate current model performance
    print("\n=== Current Model Performance Demo ===")
    if len(inputs) > 0:
        sample_input = inputs[0]
        print(f"Sample input: {sample_input}")
        
        # Tokenize input
        input_encoding = tokenizer(
            sample_input,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate with current model
        with torch.no_grad():
            outputs = model.generate(
                input_encoding["input_ids"],
                max_length=50,
                num_beams=1,
                do_sample=True,
                temperature=0.7
            )
        
        generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Current model output: {generated_query}")
    
    print("\n=== Training Approach ===")
    print("To fine-tune the T5 query rewriting model with this data:")
    print("1. Use Trainer with the prepared dataset")
    print("2. The model will learn to generate better search queries")
    print("3. During inference, generate multiple query variations")
    print("4. This improves the model's ability to find relevant sources")
    
    # Example training code (commented out as we don't want to actually train here)
    """
    # Training example (not executed):
    training_args = TrainingArguments(
        output_dir="./t5_query_rewriting_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=100,
        save_steps=1000,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    """

if __name__ == "__main__":
    demonstrate_query_rewriting_training() 