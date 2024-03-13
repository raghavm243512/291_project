from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
def tokenize_data(examples):
    inputs = tokenizer(examples['instruction'], truncation=False)
    targets = tokenizer(examples['output'], truncation=False)
    inputs["labels"] = targets["input_ids"]
    return inputs

for lang in ['eng', 'he.jap.hi_qonly', 'hi_qonly']:
    # Load the tokenizer and the model
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

    # Apply LoRA to the model
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        use_rslora=True,
        lora_alpha=32,
        lora_dropout=0.0625,
        bias='lora_only'
    )
    model = get_peft_model(model, lora_config).to(device)

    # Load the dataset
    dataset = load_dataset('parquet', data_files=f'train/{lang}.parquet')
    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=1,
        learning_rate=1e-4,
        per_device_train_batch_size=5,
        weight_decay=0.01,
        logging_dir='./logs',
        bf16=True,
        report_to='none',
        gradient_accumulation_steps=3,
        save_strategy="no"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f'models/{lang}')

    log_history = trainer.state.log_history

    # Extract the loss values
    train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()

    # Save the plot
    plt.savefig(f'loss_{lang}.png')