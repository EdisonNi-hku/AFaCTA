# AFaCTA
An automatic annotation framework for Factual Claim Detection, focusing on verifiability and calibrating with self-consistency ensemble.
One claim detector fine-tuned by full PoliClaim_gold and PoliClaim_silver can be found at https://huggingface.co/JingweiNi/roberta-base-afacta

## How to use
### How to annotate a political speech?
```shell
python code/afacta_multi_step_annotation.py --file_name data/raw_speeches/AK1995_processed.csv --output_name AK1995 --context 1 --llm_name gpt-4-0613
```
In this example command, we annotate AK1995 with a context length of 1 (previous and subsequent sentence), using GPT-4's June checkpoint.

### How to reproduce the major results in paper?
```shell
# To reproduce scores in table 3
python code/compute_scores.py

# To reproduce scores in table 6
python code/compute_scores.py --data 1

# To reproduce scores in figure 2
python code/compute_scores.py --num_answer 11 --model G3
python code/compute_scores.py --num_answer 11 --model G4

# To reproduce scores in figure 6
python code/compute_scores.py --data 1 --num_answer 11 --model G3
python code/compute_scores.py --data 1 --num_answer 11 --model G4
```

### How to train small classifiers using PoliClaim data?
```shell
python code/train_small_models.py --seed 42 --epoch 5 --base_model roberta-base --golden_data ${GOLEN_DATA_NUM} --silver_data ${SILVER_DATA_NUM} --bronze_data ${BRONZE_DATA_NUM} --batch_size 32
```
Where the mixture of golden, silver, and bronze data can be customized. For example, GOLDEN_DATA_NUM=-1 means using all golden data. GOLDEN_DATA_NUM=100 means using 100 GOLDEN data points.

## Data
We disclose all LLM generations, human annotations, and raw data in paper, which can be found in the "data" directory:

```markdown
data/
│
├── CLEF-2021_test/                       # CLEF2021 dev set
│   ├── CLEF2021_gpt_with_human_eval.xlsx # Annotations of GPT-3.5/4 and Two Human Experts
│   ├── CLEF2021_zephyr.xlsx              # Annotations of zephyr-7b
│   └── CLEF2021_llama.xlsx               # Annotations of llama-2-chat-13b
│
├── CoT_self-consistency/                 # Self-consistency CoT Generations
│   ├── clef2021_test_G3_CoT.xlsx         # GPT-3.5's generations on CLEF2021
│   ├── clef2021_test_G4_CoT.xlsx         # GPT-4's generations on CLEF2021
│   ├── policlaim_test_G3_CoT.xlsx        # GPT-3.5's generations on PoliClaim
│   ├── policlaim_test_G4_CoT.xlsx        # GPT-4's generations on PoliClaim
│
├── PoliClaim_test/                               # PoliClaim test set
│   ├── policlaim_gpt_with_human_eval_merged.xlsx # Annotations of GPT-3.5/4 and Two Human Experts, merging CA2022, AK2022, AL2022, CO2022
│   ├── policlaim_zephyr_merged.xlsx              # Annotations of zephyr-7b
│   └── policlaim_llama_merged.xlsx               # Annotations of llama-2-chat-13b
│
├── PoliClaim_train_golden/               # PoliClaim Golden training data, with human supervision (a column called "golden")
├── raw_speeches/                         # Files containing unannotated political speech data.
│   ├── ..._processed.csv                 # Sentences shorter than 30 char-length are concatenated to the previous sentences.
│   └── ....tsv                           
└── PoliClaim_train_silver_n_bronze/      # Silver and bronze training data without human double-check
```

### Fields of data files

- policlaim_gpt_with_human_eval_merged.xlsx
  - SENTENCES: target sentences
  - SPEECH: from which speech the sentence comes
  - Qx_y: annotator y's answer to Qx (x, y can be 1 or 2)
  - model_name (gpt-3.5, gpt-4, llama etc.): the aggregated AFaCTA output (0~3) from the model.
  - model_name-s?-...: output of each AFaCTA step by the model. Please refer to the paper or prompts for details about each AFaCTA score.
  - Golden: the final golden label
  - label_1/2: the label from annotator 1 or 2
- policlaim_llama_merged.xlsx
  - ver_aggregated: AFaCTA step 1 result
  - ANALYSIS1, FACT_PART1, VERIFIABLE_REASON1, VERIFIABILITY1, CATEGORY1: reasoning steps of AFaCTA step 2
  - p2_aggregated: AFaCTA step 2 result
  - subjectivity: Reasoning about not verifiable
  - objectivity: Reasoning about verifiable
  - ob_aggregated: AFaCTA step 3.1 result
  - sub_aggregated: AFaCTA step 3.2 result

Other data files have similar fields as the above two.
