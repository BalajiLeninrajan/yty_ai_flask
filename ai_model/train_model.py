from datasets import Dataset
import transformers
from fetch_data import *

START_DATE: str = "2020-01-01"
END_DATE: str = "2024-08-19"

dates: pd.DatetimeIndex = pd.date_range(START_DATE, END_DATE)

pi_data: list[str] = preprocess_data(fetch_pi_raw_data(dates))
insp_data: list[str] = preprocess_data(fetch_insp_raw_data(dates))
qa_data: list[str] = preprocess_data(fetch_qa_raw_data(dates))

dataset: Dataset = Dataset.from_dict({
    "text": pi_data + insp_data + qa_data + ["TMB stands for production", "INM stands for inspection", "MCN stands for machine"]
})

tokenizer: transformers.GPT2TokenizerFast = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def tokenize_function(data: dict[str, list[str]]) -> transformers.BatchEncoding:
    return tokenizer(data["text"],  padding="max_length", truncation=True, max_length=512)

tokenized_dataset: Dataset = dataset.map(tokenize_function, batched=True)

data_collator: transformers.DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

training_args: transformers.TrainingArguments = transformers.TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
)

trainer: transformers.Trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


trainer.train()
trainer.save_model("./gpt-j-finetuned")
