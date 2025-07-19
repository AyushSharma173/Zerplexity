# Model Fine-tuning Scripts

This folder contains scripts for fine-tuning various models using the collected training data.

## Overview

The training data is collected from user interactions in the chat interface and stored in the `../training_data/` folder. These scripts demonstrate how to use that data to fine-tune different models for improved performance.

## Available Scripts

### 1. `train_reranker.py`
**Purpose**: Fine-tune the jina-reranker-v1-turbo-en model for better source ranking

**Training Data**: Uses `../training_data/reranker_training_data.json`

**Approach**:
- Binary classification: Positive examples (user-selected top 3) vs Negative examples (remaining sources)
- Input: `[query, document_text]` pairs
- Output: Relevance score (0-1)
- Goal: Learn to rank relevant sources higher

**Usage**:
```bash
cd models_finetuning
python train_reranker.py
```

### 2. `train_query_rewriting.py`
**Purpose**: Fine-tune the T5 query rewriting model for better search query generation

**Training Data**: Uses `../training_data/query_rewriting_training_data.json`

**Approach**:
- Sequence-to-sequence: Original query → Successful query
- Input: Original user query
- Output: Query that led to selected sources
- Goal: Learn to generate queries that find relevant sources

**Usage**:
```bash
cd models_finetuning
python train_query_rewriting.py
```

## Training Data Format

### Reranker Training Data
```json
{
  "reranker_data": [
    {
      "query": "How is andrew tate's arrest in romania doing?",
      "all_sources": [
        {
          "title": "What to Know About the Accusations Against Andrew Tate - The New York Times",
          "url": "https://www.nytimes.com/article/andrew-tate-arrests-explained.html",
          "snippet": "Andrew Tate, a controversial social media influencer, was arrested in Romania..."
        }
      ],
      "positive_sources": ["url1", "url2", "url3"],
      "negative_sources": ["url4", "url5", "url6"],
      "timestamp": "2025-07-19T..."
    }
  ]
}
```

### Query Rewriting Training Data
```json
{
  "query_rewriting_data": [
    {
      "original_query": "How is andrew tate's arrest in romania doing?",
      "selected_sources": ["url1", "url2", "url3"],
      "query_source_mapping": {
        "original query": {
          "query": "original query",
          "reasoning": "Original user query",
          "sources": [...]
        },
        "rewritten query": {
          "query": "rewritten query",
          "reasoning": "T5 reformulation 1",
          "sources": [...]
        }
      },
      "query_generator": "t5",
      "timestamp": "2025-07-19T..."
    }
  ]
}
```

## Training Process

### 1. Data Collection
- Users interact with the chat interface in training mode
- When they rank sources, training data is automatically saved
- Data includes both positive and negative examples

### 2. Data Preparation
- Scripts load training data from JSON files
- Convert to appropriate format for each model type
- Create training pairs and labels

### 3. Model Fine-tuning
- Use Hugging Face Trainer or sentence-transformers
- Train on collected data
- Save fine-tuned model

### 4. Deployment
- Replace original models with fine-tuned versions
- Monitor performance improvements
- Continue collecting data for further improvements

## Model Performance

### Reranker Model
- **Base Model**: jina-reranker-v1-turbo-en (37.8M parameters)
- **Input**: Query-document pairs
- **Output**: Relevance scores (0-1)
- **Goal**: Rank relevant sources higher

### Query Rewriting Model
- **Base Model**: prhegde/t5-query-reformulation-RL
- **Input**: Original user query
- **Output**: Multiple search query variations
- **Goal**: Generate queries that find relevant sources

## Usage Examples

### Running Training Demonstrations
```bash
# Show reranker training data and approach
python train_reranker.py

# Show query rewriting training data and approach
python train_query_rewriting.py
```

### Actual Training (Uncomment in scripts)
```python
# In train_reranker.py
model.fit(
    train_pairs=train_pairs,
    train_labels=labels,
    epochs=3,
    batch_size=16,
    warmup_steps=100,
    show_progress_bar=True
)

# In train_query_rewriting.py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Data Collection Strategy

### User-Friendly Approach
- Users only need to select top 3 sources (not rank every source)
- System automatically treats remaining sources as negative examples
- Creates comprehensive training signal without user burden

### Continuous Learning
- Training data accumulates over time
- Models can be periodically re-fine-tuned
- Performance improves with more user interactions

## Next Steps

1. **Collect More Data**: Run queries in training mode to gather user feedback
2. **Fine-tune Models**: Use collected data to improve model performance
3. **Deploy Fine-tuned Models**: Replace base models with improved versions
4. **Monitor Performance**: Track improvements in source relevance and query generation
5. **Iterate**: Continue collecting data and re-training models

## File Structure
```
models_finetuning/
├── README.md                    # This file
├── train_reranker.py           # Reranker fine-tuning script
└── train_query_rewriting.py    # Query rewriting fine-tuning script

../training_data/
├── training_data.json          # Model comparison training data
├── query_rewriting_training_data.json  # Query rewriting training data
└── reranker_training_data.json # Reranker training data
``` 