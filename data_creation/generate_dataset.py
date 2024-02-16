import pandas as pd
import sqlparse
import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

data = pd.read_parquet('cleaned.parquet')
batch_size = 64
size = len(data)
target_lang = 'hi_IN'
output = pd.DataFrame(columns=['instruction', 'output'])
split_dataset = False
secondary_langs = ['ko_KR', 'tr_TR']
output_name = f'{target_lang}.{secondary_langs[0]}.{secondary_langs[1]}.parquet' if split_dataset else f'{target_lang}.parquet'

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


def translate(batch): # Convert batch of strings into target language
    tokenizer.src_lang = "en_XX"
    encoded_words = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    generated_tokens = model.generate(
        **encoded_words,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations


id_map = {}
translation_batch = []
# Context is the same every time
context_translation = data.loc[0]['instruction'].split('--')[1]
context_translation = translate([context_translation])

for index, row in data.iterrows():
    if split_dataset:
        if index > size // 3:
            target_lang = secondary_langs[0]
        elif index > size * 2 // 3:
            target_lang = secondary_langs[1]
            split_dataset = False
    instruction, output = row['instruction'], row['output']
    # Separate instruction with known format
    parts = instruction.split('--')
    sql = parts[0]
    question = parts[2]

    # Convert SQL to extract translatable items
    tokens = list(sqlparse.parse(sql)[0].flatten())
    identifiers = [token.value for token in tokens if token.ttype.__repr__() in ['Token.Name', 'Token.Literal.String.Symbol']]

    for idf in identifiers:
        if idf in id_map or idf in translation_batch: # leverage old knowledge (some things may be common)
            continue
        translation_batch.append(idf)

        # Translate the batch and save results to map
        if len(translation_batch) == batch_size:
            translated = translate(translation_batch)
            for raw, trans in zip(translation_batch, translated):
                id_map[raw] = trans if '"' in trans else f'"{trans}"' # Quote special characters from translation so they are now symbols
            translation_batch.clear()
    
    # Use question with remaining batch elements or potentially its own batch
    translation_batch.append(question)
    translated = translate(translation_batch)

    # Exclude question from id map
    for i in range(len(translated)-1):
        id_map[translation_batch[i]] = translated[i] if '"' in translated[i] else f'"{translated[i]}"'

    # Replace all identifiers in sql and output sql
    for idf in identifiers:
        sql = re.sub(fr'(\b|\.){idf}(\b|\.)', id_map[idf], sql)
        output = re.sub(fr'(\b|\.){idf}(\b|\.)', id_map[idf], output)
    
    translation_batch.clear()

    # Create new instruction
    instruction = '--'.join((sql, context_translation, translated[-1]))
    
    # Clear map if too big
    if len(id_map) > 5000:
        id_map.clear()
    
    output = output.append({'instruction': instruction, 'output': output})

output.to_parquet(output_name)