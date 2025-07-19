# Training Data

This folder contains all training data collected from user interactions with the chat interface.

## Overview

The training data is automatically collected when users interact with the system in training mode. This data is used to fine-tune various models for improved performance.

## Data Files

### 1. `training_data.json`
**Purpose**: Model comparison training data

**Content**:
- User queries
- Two model responses (GPT-4o vs Llama) with different temperatures
- User's selected "better" response
- Top 3 ranked sources

**Use Case**: Training for model comparison and response quality evaluation

**Format**:
```json
{
  "training_data": [
    {
      "id": "training_20250719_133509_0",
      "query": "How is andrew tate's arrest in romania doing?",
      "response1": {
        "text": "Andrew Tate, a controversial social media influencer...",
        "temperature": 0.3,
        "latency": 15.471,
        "tokens_per_minute": 779.5
      },
      "response2": {
        "text": "Andrew Tate, a controversial social media influencer...",
        "temperature": 0.7,
        "latency": 15.471,
        "tokens_per_minute": 779.5
      },
      "selected_response": 1,
      "sources": [
        {
          "title": "What to Know About the Accusations Against Andrew Tate - The New York Times",
          "url": "https://www.nytimes.com/article/andrew-tate-arrests-explained.html",
          "ranked": true
        }
      ],
      "timestamp": "2025-07-19T13:35:09.592486"
    }
  ]
}
```

### 2. `query_rewriting_training_data.json`
**Purpose**: Query rewriting model training data

**Content**:
- Original user queries
- Generated query variations (GPT-4o or T5)
- Which queries led to selected sources
- Query-source mappings

**Use Case**: Fine-tuning query generation models to produce better search queries

**Format**:
```json
{
  "query_rewriting_data": [
    {
      "original_query": "How is andrew tate's arrest in romania doing?",
      "selected_sources": [
        "https://www.nytimes.com/article/andrew-tate-arrests-explained.html",
        "https://www.nbcnews.com/news/world/andrew-tate-tristan-romania-rape-trafficking-travel-ban-us-rcna193959"
      ],
      "query_source_mapping": {
        "How is andrew tate's arrest in romania doing?": {
          "query": "How is andrew tate's arrest in romania doing?",
          "reasoning": "Original user query",
          "sources": [...]
        },
        "which was the prime focus for andrew tate's arrest in romania?": {
          "query": "which was the prime focus for andrew tate's arrest in romania?",
          "reasoning": "T5 reformulation 1",
          "sources": [...]
        }
      },
      "query_generator": "t5",
      "timestamp": "2025-07-19T18:35:19.139Z"
    }
  ]
}
```

### 3. `reranker_training_data.json`
**Purpose**: Reranker model training data

**Content**:
- All sources from all search queries (consolidated)
- User's top 3 selected sources (positive examples)
- Remaining sources (negative examples)

**Use Case**: Fine-tuning the reranker model to better rank relevant sources

**Format**:
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
      "positive_sources": [
        "https://www.nytimes.com/article/andrew-tate-arrests-explained.html",
        "https://www.nbcnews.com/news/world/andrew-tate-tristan-romania-rape-trafficking-travel-ban-us-rcna193959"
      ],
      "negative_sources": [
        "https://unrelated-news.com/andrew-tate-controversy",
        "https://visit-romania.com/tourism-guide"
      ],
      "timestamp": "2025-07-19T..."
    }
  ]
}
```

## Data Collection Process

### 1. User Interaction
- User asks a question in training mode
- System generates multiple search queries
- System retrieves sources from all queries
- System generates two model responses for comparison

### 2. User Feedback
- User selects the "better" response
- User ranks top 3 most relevant sources
- System automatically saves training data

### 3. Data Storage
- Training data is saved to appropriate JSON files
- Data includes both positive and negative examples
- Timestamps track when data was collected

## Data Usage

### Model Comparison Training
- Used to evaluate which model responses are preferred
- Helps tune model parameters (temperature, etc.)
- Provides insights into user preferences

### Query Rewriting Training
- Used to fine-tune query generation models
- Learns which query variations lead to better sources
- Improves search query generation

### Reranker Training
- Used to fine-tune the source ranking model
- Learns to rank relevant sources higher
- Improves source selection for context building

## Data Quality

### Positive Examples
- User-selected top 3 sources
- High-quality, relevant sources
- Used as positive training examples

### Negative Examples
- Remaining sources not selected by user
- Lower-quality or less relevant sources
- Used as negative training examples

### Data Balance
- Each query provides multiple training examples
- Positive/negative ratio varies by query
- System automatically balances training data

## File Management

### Automatic Creation
- Files are created automatically when first data is collected
- New entries are appended to existing files
- No manual intervention required

### Data Persistence
- Data persists across application restarts
- Files are stored in JSON format for easy inspection
- Can be backed up or version controlled

### Data Size
- Files grow as more user interactions occur
- Each entry includes comprehensive metadata
- Efficient storage format for training purposes

## Next Steps

1. **Collect Data**: Run queries in training mode to gather user feedback
2. **Analyze Data**: Review collected data for quality and patterns
3. **Train Models**: Use data to fine-tune models (see `../models_finetuning/`)
4. **Deploy Models**: Replace base models with fine-tuned versions
5. **Monitor Performance**: Track improvements in model performance
6. **Iterate**: Continue collecting data and re-training models

## File Structure
```
training_data/
├── README.md                           # This file
├── training_data.json                  # Model comparison training data
├── query_rewriting_training_data.json  # Query rewriting training data
└── reranker_training_data.json        # Reranker training data
``` 