# train_sft.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

MODEL_NAME = "prhegde/t5-query-reformulation-RL"  # or "t5-base"
MAX_IN  = 128
MAX_OUT = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

ds = load_dataset("json", data_files={
    "train": "qr_train.jsonl",
    "validation": "qr_valid.jsonl"
})

def preprocess(batch):
    model_inputs = tokenizer(batch["input"], max_length=MAX_IN, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target"], max_length=MAX_OUT, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(preprocess, batched=True, remove_columns=["input","target"])

args = Seq2SeqTrainingArguments(
    output_dir="qr_t5_sft",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=MAX_OUT,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none"
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
)

trainer.train()
trainer.save_model("qr_t5_sft/final")
tokenizer.save_pretrained("qr_t5_sft/final")
