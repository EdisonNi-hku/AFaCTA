import pandas as pd
import os
import glob
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from transformers.utils import is_torch_available


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_test(tokenizer, args):
    policlaim_1 = pd.read_excel('data/PoliClaim_test/AL2022_human_eval.xlsx')
    policlaim_2 = pd.read_excel('data/PoliClaim_test/AK2022_human_eval.xlsx')
    policlaim_3 = pd.read_excel('data/PoliClaim_test/CO2022_human_eval.xlsx')
    policlaim_4 = pd.read_excel('data/PoliClaim_test/CA2022_human_eval.xlsx')
    checkthat = pd.read_excel('data/CLEF-2021_test/CLEF2021_gpt_with_human_eval.xlsx')


    parsed_sents = []
    test_label_1 = []
    for policlaim in [policlaim_1, policlaim_2, policlaim_3, policlaim_4]:
        test_sent, test_label = policlaim['SENTENCES'].to_list(), policlaim['Golden'].to_list()
        test_label_1 += test_label
        for i, sent in enumerate(test_sent):
            if not args.no_context:
                if i == 0:
                    parsed_sents.append(test_sent[i] + tokenizer.sep_token + test_sent[i + 1])
                elif i == len(test_sent) - 1:
                    parsed_sents.append(test_sent[i - 1] + tokenizer.sep_token + test_sent[i])
                else:
                    parsed_sents.append(
                        test_sent[i - 1] + tokenizer.sep_token + test_sent[i] + tokenizer.sep_token + test_sent[i + 1])
            else:
                parsed_sents.append(sent)

    test_sent_2, test_label_2 = checkthat['SENTENCES'].to_list(), checkthat['Golden'].to_list()

    return parsed_sents, test_sent_2, test_label_1, test_label_2


def train_data_mix(tokenizer, args):
    golden_sents, golden_labels, silver_sents, silver_labels, bronze_sents, bronze_labels = load_train(tokenizer, args)

    if args.golden_data >= 0:
        golden_indices = random.sample(range(len(golden_sents)), args.golden_data)
        sampled_golden_sents = [golden_sents[i] for i in golden_indices]
        sampled_golden_labels = [golden_labels[i] for i in golden_indices]
    else:
        sampled_golden_sents = golden_sents
        sampled_golden_labels = golden_labels

    if args.silver_data >= 0:
        silver_indices = random.sample(range(len(silver_sents)), args.silver_data)
        sampled_silver_sents = [silver_sents[i] for i in silver_indices]
        sampled_silver_labels = [silver_labels[i] for i in silver_indices]
    else:
        sampled_silver_sents = silver_sents
        sampled_silver_labels = silver_labels

    if args.bronze_data >= 0:
        bronze_indices = random.sample(range(len(bronze_sents)), args.bronze_data)
        sampled_bronze_sents = [bronze_sents[i] for i in bronze_indices]
        sampled_bronze_labels = [bronze_labels[i] for i in bronze_indices]
    else:
        sampled_bronze_sents = bronze_sents
        sampled_bronze_labels = bronze_labels

    print(type(sampled_golden_sents))
    print(type(sampled_silver_sents))
    print(type(sampled_bronze_sents))

    train_sents = sampled_golden_sents + sampled_silver_sents + sampled_bronze_sents
    train_labels = sampled_golden_labels + sampled_silver_labels + sampled_bronze_labels

    return train_sents, train_labels


def load_train(tokenizer, args):
    golden_files = glob.glob(os.path.join(args.train_golden, '*_1.xlsx'))
    golden_sents = []
    golden_labels = []
    consistent_len = 0
    for file in golden_files:
        df = pd.read_excel(file)
        sents = df['SENTENCES'].to_list()
        labels = df['golden'].to_list()
        consistent_len += len(df.loc[df['likelihood'].isin([0, 3]), :])
        parsed_sents = []
        for i, sent in enumerate(sents):
            if not args.no_context:
                if i == 0:
                    parsed_sents.append(sents[i] + tokenizer.sep_token + sents[i + 1])
                elif i == len(sents) - 1:
                    parsed_sents.append(sents[i - 1] + tokenizer.sep_token + sents[i])
                else:
                    parsed_sents.append(sents[i - 1] + tokenizer.sep_token + sents[i] + tokenizer.sep_token + sents[i + 1])
            else:
                parsed_sents.append(sent)
        golden_sents += parsed_sents
        golden_labels += labels
    print(consistent_len)

    silver_files = glob.glob(os.path.join(args.train_silver, '*_1.xlsx'))
    silver_labels = []
    silver_sents = []
    silver_consistencies = []
    for file in silver_files:
        df = pd.read_excel(file)
        sents = df['SENTENCES']
        labels = [0 if l <= 1.5 else 1 for l in df['likelihood']]
        consistency = [1 if l == 0 or l == 3 else 0 for l in df['likelihood']]
        parsed_sents = []
        for i, sent in enumerate(sents):
            if not args.no_context:
                if i == 0:
                    parsed_sents.append(sents[i] + tokenizer.sep_token + sents[i + 1])
                elif i == len(sents) - 1:
                    parsed_sents.append(sents[i - 1] + tokenizer.sep_token + sents[i])
                else:
                    parsed_sents.append(sents[i - 1] + tokenizer.sep_token + sents[i] + tokenizer.sep_token + sents[i + 1])
            else:
                parsed_sents.append(sent)
        silver_sents += parsed_sents
        silver_labels += labels
        silver_consistencies += consistency

    final_silver = []
    final_silver_labels = []
    final_bronze = []
    final_bronze_labels = []

    for sent, label, consist in zip(silver_sents, silver_labels, silver_consistencies):
        if consist == 0:
            final_bronze.append(sent)
            final_bronze_labels.append(label)
        else:
            final_silver.append(sent)
            final_silver_labels.append(label)

    print('bronze', len(final_bronze))
    print('silver', len(final_silver))
    print('golden', len(golden_sents))

    print('bronze claim', sum(final_bronze_labels))
    print('silver claim', sum(final_silver_labels))
    print('golden claim', sum(golden_labels))

    return golden_sents, golden_labels, final_silver, final_silver_labels, final_bronze, final_bronze_labels


def evaluate_model(args, model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        labels = batch['labels']
        all_labels += labels.cpu().tolist()
        all_preds += predictions.cpu().tolist()

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    train_sent, train_label = train_data_mix(tokenizer, args)
    test_sent_1, test_sent_2, test_label_1, test_label_2 = load_test(tokenizer, args)

    train_dataset = TextDataset(tokenizer, train_sent, train_label)
    test_dataset1 = TextDataset(tokenizer, test_sent_1, test_label_1)
    test_dataset2 = TextDataset(tokenizer, test_sent_2, test_label_2)

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, cache_dir=args.cache_dir, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_dir,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gas,
        evaluation_strategy="no",
        save_strategy='no',
        load_best_model_at_end=False,
        report_to="none",
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()
    model.save_pretrained(args.model_save_dir)

    # Calculate accuracy on testset1 and testset2

    acc = evaluate_model(args, model, test_dataset1)

    print("Acc", acc)
    print("---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_golden", type=str, default="data/PoliClaim_train_golden")
    parser.add_argument("--train_silver", type=str, default="data/PoliClaim_train_silver_n_bronze")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--golden_data", type=int, default=0)
    parser.add_argument("--silver_data", type=int, default=0)
    parser.add_argument("--bronze_data", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="distilroberta-base")
    parser.add_argument("--cache_dir", type=str, default="/home/jini/shares/transformer_models")
    parser.add_argument("--model_save_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gas", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_context", action='store_true', default=False)
    parser.add_argument("--special_arch", action='store_true', default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
