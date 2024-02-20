import os
import pandas as pd

dir_path = 'data/raw_speeches'
file_list = os.listdir(dir_path)

for filename in file_list:
    if filename.endswith(".tsv"):
        tsv_file_path = os.path.join(dir_path, filename)
        df = pd.read_csv(tsv_file_path, sep='\t')
        sentences = df.sentences.to_list()
        parsed_sentences = []
        for s in sentences:
            if len(s) > 30 or len(parsed_sentences) == 0:
                parsed_sentences.append(s)
            else:
                parsed_sentences[-1] += s
        pd.DataFrame({'SENTENCES': parsed_sentences}).to_csv(tsv_file_path.replace('.tsv', '_processed.csv'), index=False)


