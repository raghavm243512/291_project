import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import pandas as pd
import sqlparse
import re
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

batch_size = 140
target_lang = 'he'
output_frame = pd.DataFrame(columns=['instruction', 'output'])
split_dataset = True
secondary_langs = ['jap', 'hi']
output_name = f'{target_lang}.{secondary_langs[0]}.{secondary_langs[1]}.parquet' if split_dataset else f'{target_lang}.parquet'

model, tokenizer = None, None
def create_model():
    global model, tokenizer, target_lang
    if model is not None:
        del model
        del tokenizer
    model_name = f'Helsinki-NLP/opus-mt-en-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name).to('cuda')
    tokenizer = MarianTokenizer.from_pretrained(model_name)

create_model()

data = pd.read_parquet('eng.parquet')
size = len(data)

def translate(batch): # Convert batch of strings into target language
    encoded_words = tokenizer.prepare_seq2seq_batch(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        generated_tokens = model.generate(**encoded_words).cpu()
        del encoded_words
    torch.cuda.empty_cache()  # Free up GPU memory
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations


translation_batch = set()
translation_buffer = set()
# Context is the same every time
context = data.loc[0]['instruction'].split('--')[1]
context_translation = translate([context])[0]

instructions = []
outputs = []
idf_to_instr = {}
questions = []

def insert_data(translation_batch, instructions, outputs, questions, idf_to_instr):
    global output_frame
    batch = sorted(list(translation_batch), key=len, reverse=True) # order by longest to prevent substring replacement
    translated = [re.escape(t) if '"' in t or "'" in t else f'"{re.escape(t)}"' for t in translate(batch)]
    for orig, trans in zip(batch, translated):
        for i in idf_to_instr[orig]:
            if questions[i] == orig: # replace question at end, known format
                instructions[i] = '--'.join((instructions[i].split('--')[0], context_translation, trans[1:-1] if trans[0] == '"' and trans[-1] == '"' else trans))
            else:
                orig = re.escape(orig)
                instructions[i] = re.sub(fr'{orig}', trans, instructions[i])
                outputs[i] = re.sub(fr'{orig}', trans, outputs[i])
    temp_frame = pd.DataFrame({'instruction': instructions, 'output': outputs})
    output_frame = pd.concat([output_frame, temp_frame], ignore_index=True)
    

for index, row in tqdm(data.iterrows(), total=size):
    if split_dataset:
        if index == size // 3:
            target_lang = secondary_langs[0]
            create_model()
            context_translation = translate([context])[0]
        elif index == size * 2 // 3:
            target_lang = secondary_langs[1]
            create_model()
            context_translation = translate([context])[0]
            split_dataset = False
    instruction, output = row['instruction'], row['output']
    # Separate instruction with known format
    parts = instruction.split('--')
    sql = parts[0]
    question = parts[2]

    # Convert SQL to extract translatable items
    statements = sqlparse.parse(sql)
    tokens = []
    for s in statements: # Multiple statements, combine into one big flattened list
        tokens += list(s.flatten())

    statements = sqlparse.parse(output)
    for s in statements: # Multiple statements, combine into one big flattened list
        tokens += list(s.flatten())
    
    identifiers = [token.value for token in tokens if token.ttype.__repr__() in ['Token.Name', 'Token.Literal.String.Symbol', 'Token.Literal.String.Single']]

    for idf in identifiers:
        if idf in translation_batch: # leverage old knowledge (some things may be common)
            continue
        translation_buffer.add(idf)
    
    translation_buffer.add(question)

    if len(translation_buffer) + len(translation_batch) > batch_size: # If buffer can't be added to batch
        if len(translation_batch) > 0: # non empty batch
            insert_data(translation_batch, instructions, outputs, questions, idf_to_instr)
            translation_batch.clear()
            instructions.clear()
            questions.clear()
            outputs.clear()
            idf_to_instr.clear()

        temp_buffer = []
        if len(translation_buffer) > batch_size: # buffer larger than a batch
            temp_buffer = sorted(list(translation_buffer), key=len, reverse=True)
            while len(temp_buffer) > batch_size: # process buffer elements until it fits in a batch
                batch, temp_buffer = temp_buffer[:batch_size], temp_buffer[batch_size:]
                translated = [re.escape(t) if '"' in t or "'" in t else f'"{re.escape(t)}"' for t in translate(batch)]
                for orig, trans in zip(batch, translated):
                    if question == orig: # replace question at end, known format
                        instruction = '--'.join((instruction.split('--')[0], context_translation, trans[1:-1] if trans[0] == '"' and trans[-1] == '"' else trans))
                    else:
                        orig = re.escape(orig)
                        instruction = re.sub(fr'{orig}', trans, instruction)
                        output = re.sub(fr'{orig}', trans, output)
            translation_buffer = set(temp_buffer)
    
    translation_batch = translation_buffer.union(translation_batch)
    for t in translation_buffer:
        if t in idf_to_instr:
            idf_to_instr[t].append(len(instructions))
        else:
            idf_to_instr[t] = [len(instructions)]
    translation_buffer.clear()
    instructions.append(instruction)
    outputs.append(output)
    questions.append(question)

# There will always be a partial or full batch remaining with 1 or more instructions
insert_data(translation_batch, instructions, outputs, questions, idf_to_instr)

output_frame.to_parquet(output_name)