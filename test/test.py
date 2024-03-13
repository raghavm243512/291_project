from transformers import T5Tokenizer
import torch
from peft import AutoPeftModelForSeq2SeqLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

langs = ['eng', 'he.jap.hi_qonly', 'hi_qonly']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
loss_df = pd.DataFrame(columns=['model', 'dataset', 'loss'])

with torch.no_grad():
    for mlang in langs:
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(f'models/{mlang}').to(device)
        model.eval()

        for dlang in tqdm(langs, total=len(langs)):
            dataset = load_dataset('parquet', data_files=f'test/{dlang}_test.parquet')
            dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=False)  # Adjust batch size as needed

            total_loss = 0
            output_df = pd.DataFrame(columns=['model_output'])

            for batch in dataloader:
                inputs = tokenizer(batch['instruction'], return_tensors='pt', padding=True, truncation=False).input_ids.to(device)
                targets = tokenizer(batch['output'], return_tensors='pt', padding=True, truncation=False).input_ids.to(device)

                decoder_input_ids = torch.cat([
                    torch.ones((targets.size(0), 1), device=device, dtype=torch.long) * tokenizer.pad_token_id,
                    targets[:, :-1]
                ], dim=-1)
                # Forward pass
                outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits

                # Compute loss
                loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()

                # Detokenize model outputs
                model_outputs = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)

                # Save model outputs
                batch_df = pd.DataFrame({'model_output': model_outputs})
                output_df = pd.concat([output_df, batch_df], ignore_index=True)

            # Compute average loss
            avg_loss = total_loss / len(dataloader)
            dataset_df = pd.DataFrame({'model': [mlang], 'dataset': [dlang], 'loss': [avg_loss]})
            loss_df = pd.concat([loss_df, dataset_df], ignore_index=True)

            # Save model outputs to a Parquet file
            output_df.to_parquet(f'model_outputs/{mlang}_{dlang}_outputs.parquet', index=False)

    # Save losses to a CSV file
    loss_df.to_csv('model_losses.csv', index=False)