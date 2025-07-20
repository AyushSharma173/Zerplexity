# storage.py
import json
import os
import boto3
import io
from pathlib import Path

S3_BUCKET = os.getenv("TRAINING_S3_BUCKET")

# Initialize S3 client (will be None if no bucket configured)
s3 = boto3.client("s3") if S3_BUCKET else None

def load_json(key, default):
    """Load JSON data from S3 or return default"""
    if not s3:
        # Fallback to local file system
        file_path = Path(key)
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading local file {key}: {e}")
        return default
    
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.load(obj['Body'])
    except s3.exceptions.NoSuchKey:
        return default
    except Exception as e:
        print(f"Error loading from S3 {key}: {e}")
        return default

def save_json(key, data):
    """Save JSON data to S3 or local file system"""
    if not s3:
        # Fallback to local file system
        file_path = Path(key)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to local file: {key}")
        except Exception as e:
            print(f"Error saving to local file {key}: {e}")
        return
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET, 
            Key=key, 
            Body=json.dumps(data).encode("utf-8"), 
            ContentType="application/json"
        )
        print(f"Saved to S3: {key}")
    except Exception as e:
        print(f"Error saving to S3 {key}: {e}") 