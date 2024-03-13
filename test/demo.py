import gradio as gr
from peft import AutoPeftModelForSeq2SeqLM
from transformers import T5Tokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('models/tokenizer')
model = AutoPeftModelForSeq2SeqLM.from_pretrained('models/eng').to(device)
model.eval()

def infer(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").input_ids
        decoder_input_ids = torch.ones((inputs.size(0), 1), dtype=torch.long) * tokenizer.pad_token_id
        inputs = inputs.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        for _ in range(512):
            outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

            if torch.any(next_token_id == tokenizer.eos_token_id):
                break

        res = tokenizer.batch_decode(decoder_input_ids.cpu(), skip_special_tokens=True)
        return res[0]


demo = gr.Interface(
    fn=infer,
    inputs=["text"],
    outputs=["text"],
)

demo.launch(share=True)