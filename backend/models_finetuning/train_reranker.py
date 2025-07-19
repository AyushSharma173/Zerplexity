#!/usr/bin/env python3
"""
Script to demonstrate how to use collected training data to fine-tune the reranker model.
This shows the data format and training approach for the jina-reranker-v1-turbo-en model.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RerankerTrainingDataset(Dataset):
    """Dataset for reranker training from collected data"""
    
    def __init__(self, training_data: List[Dict], max_length: int = 512):
        self.max_length = max_length
        self.pairs = []
        self.labels = []
        
        # Process training data into query-document pairs
        for entry in training_data:
            query = entry["query"]
            all_sources = entry["all_sources"]
            positive_urls = set(entry["positive_sources"])
            negative_urls = set(entry["negative_sources"])
            
            # Create positive pairs (query + positive document)
            for source in all_sources:
                if source["url"] in positive_urls:
                    # Create document text from title and snippet
                    doc_text = f"{source['title']} {source.get('snippet', '')}"
                    self.pairs.append([query, doc_text])
                    self.labels.append(1.0)  # Positive example
                
                elif source["url"] in negative_urls:
                    # Create document text from title and snippet
                    doc_text = f"{source['title']} {source.get('snippet', '')}"
                    self.pairs.append([query, doc_text])
                    self.labels.append(0.0)  # Negative example
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

def load_training_data() -> List[Dict]:
    """Load reranker training data from file"""
    training_file = Path("../training_data/reranker_training_data.json")
    if not training_file.exists():
        print("No reranker training data found. Run some queries in training mode first.")
        return []
    
    with open(training_file, 'r') as f:
        data = json.load(f)
        return data.get("reranker_data", [])

def prepare_training_data(training_data: List[Dict]) -> Tuple[List, List]:
    """Prepare training data for the reranker model"""
    print(f"Processing {len(training_data)} training entries...")
    
    queries = []
    documents = []
    labels = []
    
    for entry in training_data:
        query = entry["query"]
        all_sources = entry["all_sources"]
        positive_urls = set(entry["positive_sources"])
        negative_urls = set(entry["negative_sources"])
        
        print(f"Query: {query}")
        print(f"  Positive sources: {len(positive_urls)}")
        print(f"  Negative sources: {len(negative_urls)}")
        
        for source in all_sources:
            # Create document text from title and snippet
            doc_text = f"{source['title']} {source.get('snippet', '')}"
            queries.append(query)
            documents.append(doc_text)
            
            # Label: 1 for positive, 0 for negative
            if source["url"] in positive_urls:
                labels.append(1.0)
            elif source["url"] in negative_urls:
                labels.append(0.0)
            else:
                # Skip unranked sources for now
                continue
    
    print(f"Total training pairs: {len(queries)}")
    print(f"Positive examples: {sum(labels)}")
    print(f"Negative examples: {len(labels) - sum(labels)}")
    
    return queries, documents, labels

def demonstrate_reranker_training():
    """Demonstrate how to use the training data for reranker fine-tuning"""
    
    # Load training data
    training_data = load_training_data()
    if not training_data:
        print("No training data available. Please run some queries in training mode first.")
        return
    
    # Prepare training data
    queries, documents, labels = prepare_training_data(training_data)
    
    if len(queries) == 0:
        print("No valid training pairs found.")
        return
    
    # Load the reranker model
    print("Loading reranker model...")
    model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)
    
    # Create training pairs
    train_pairs = [[query, doc] for query, doc in zip(queries, documents)]
    
    print("\n=== Training Data Preview ===")
    print("Sample positive examples:")
    positive_indices = [i for i, label in enumerate(labels) if label == 1.0][:3]
    for idx in positive_indices:
        print(f"  Query: {queries[idx]}")
        print(f"  Document: {documents[idx][:100]}...")
        print(f"  Label: {labels[idx]}")
        print()
    
    print("Sample negative examples:")
    negative_indices = [i for i, label in enumerate(labels) if label == 0.0][:3]
    for idx in negative_indices:
        print(f"  Query: {queries[idx]}")
        print(f"  Document: {documents[idx][:100]}...")
        print(f"  Label: {labels[idx]}")
        print()
    
    # Demonstrate ranking with current model
    print("\n=== Current Model Ranking Demo ===")
    if len(train_pairs) > 0:
        # Take a sample query and its documents
        sample_query = queries[0]
        sample_docs = [doc for i, doc in enumerate(documents) if queries[i] == sample_query][:5]
        
        print(f"Sample query: {sample_query}")
        print("Ranking documents...")
        
        # Create pairs for ranking
        ranking_pairs = [[sample_query, doc] for doc in sample_docs]
        
        # Get scores from current model
        scores = model.predict(ranking_pairs)
        
        # Sort by score
        ranked_results = sorted(zip(sample_docs, scores), key=lambda x: x[1], reverse=True)
        
        print("Current model ranking:")
        for i, (doc, score) in enumerate(ranked_results):
            print(f"  {i+1}. Score: {score:.3f} | {doc[:80]}...")
    
    print("\n=== Training Approach ===")
    print("To fine-tune the reranker with this data:")
    print("1. Use CrossEncoder.fit() with the training pairs and labels")
    print("2. The model will learn to assign higher scores to positive examples")
    print("3. During inference, rank documents by their predicted scores")
    print("4. This improves the model's ability to identify relevant sources")
    
    # Example training code (commented out as we don't want to actually train here)
    """
    # Training example (not executed):
    model.fit(
        train_pairs=train_pairs,
        train_labels=labels,
        epochs=3,
        batch_size=16,
        warmup_steps=100,
        show_progress_bar=True
    )
    """

if __name__ == "__main__":
    demonstrate_reranker_training() 