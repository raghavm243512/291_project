import pandas as pd
import sqlglot

def parse(output, model_output):
    try:
        # Parse the SQL queries to ASTs
        output_ast = sqlglot.parse(output, read='sqlite')[0]
        model_output_ast = sqlglot.parse(model_output, read='sqlite')[0]

        # Convert the ASTs to trees for comparison
        diffs = sqlglot.diff(output_ast, model_output_ast)

        # If parsing is successful, return 1 and the normalized edit distance
        return 1, len(diffs) # denominator needed
    except Exception:
        # If parsing fails, return 0 and None
        return 0, None


langs = ['eng', 'he.jap.hi_qonly', 'hi_qonly', 'es']
results = []

for dlang in langs:
    gt = pd.read_parquet(f'data/test/{dlang}_test.parquet')
    for mlang in ['base'] + langs[:-1]:
        output = pd.read_parquet(f'model_outputs/{mlang}_{dlang}_outputs.parquet')
        df = pd.concat([gt, output], axis=1)
        df['parse_result'], df['edit_distance'] = zip(*df.apply(lambda row: parse(row['output'], row['model_output']), axis=1))
        parse_success_rate = df['parse_result'].mean()
        avg_edit_distance = df['edit_distance'].dropna().mean()
        results.append((mlang, dlang, parse_success_rate, avg_edit_distance))

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(sorted(results), columns=['mlang', 'dlang', 'parse_success_rate', 'avg_edit_distance'])
results_df.to_csv('model_asts.csv', index=False)