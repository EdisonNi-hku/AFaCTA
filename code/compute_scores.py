import pandas as pd
import argparse
import random
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

random.seed(42)


def compute_likelihood(df_to_eval):
    p1 = df_to_eval['gpt-4-s1'].apply(lambda x: 1 if "yes" in x.lower() else 0).values
    p2 = []
    for i, c in zip(df_to_eval['gpt-4-s2'], df_to_eval['gpt-4-category']):
        if i or 'C0' not in c:
            p2.append(1)
        else:
            p2.append(0)
    p2 = np.array(p2)
    p3_1 = df_to_eval['gpt-4-s3-1'].apply(lambda x: 1 if "objective" in x.lower() else 0).values
    p3_2 = df_to_eval['gpt-4-s3-2'].apply(lambda x: 1 if "objective" in x.lower() else 0).values

    df_to_eval['gpt-4'] = p1 + p2 + 0.5 * p3_1 + 0.5 * p3_2

    p1 = df_to_eval['gpt-3.5-s1'].apply(lambda x: 1 if "yes" in x.lower() else 0).values
    p2 = []
    for i, c in zip(df_to_eval['gpt-3.5-s2'], df_to_eval['gpt-3.5-category']):
        if i or 'C0' not in c:
            p2.append(1)
        else:
            p2.append(0)
    p2 = np.array(p2)
    p3_1 = df_to_eval['gpt-3.5-s3-1'].apply(lambda x: 1 if "objective" in x.lower() else 0).values
    p3_2 = df_to_eval['gpt-3.5-s3-2'].apply(lambda x: 1 if "objective" in x.lower() else 0).values

    df_to_eval['gpt-3.5'] = p1 + p2 + 0.5 * p3_1 + 0.5 * p3_2
    return df_to_eval


def mapping(l, neg):
    return [0 if i == neg or str(neg).lower() in str(i).lower() else 1 for i in l]


def mapping_two(l1, l2, neg):
    ret = []
    for i1, i2 in zip(l1, l2):
        if i1 != i2:
            ret.append(random.randint(0, 1))
        elif i1 == neg:
            ret.append(0)
        else:
            ret.append(1)
    return ret


def main(df):
    df = compute_likelihood(df)

    df['gpt35_label'] = df['gpt-3.5'].apply(lambda x: 0 if x <= 1.5 else 1)
    df['gpt4_label'] = df['gpt-4'].apply(lambda x: 0 if x <= 1.5 else 1)
    df['zephyr_label'] = df['zephyr'].apply(lambda x: 0 if x <= 1.5 else 1)
    df['llama_label'] = df['llama'].apply(lambda x: 0 if x <= 1.5 else 1)

    print(np.mean(df['Golden'])) # 81.43, 63.85
    print("===zephyr===")
    print('Acc', accuracy_score(df['Golden'], df['zephyr_label']))
    print('kappa', (cohen_kappa_score(df['label_1'], df['zephyr_label']) + cohen_kappa_score(df['label_2'], df['zephyr_label']) / 2))

    print("===llama===")
    print('Acc', accuracy_score(df['Golden'], df['llama_label']))
    print('kappa', (cohen_kappa_score(df['label_1'], df['llama_label']) + cohen_kappa_score(df['label_2'],
                                                                                             df['llama_label']) / 2))

    print("===GPT-4===")
    #sub_df = df.loc[(df['gpt-4'] < 1) | (df['gpt-4'] > 1.5), :]
    sub_df = df
    print("S1 label")
    print('kappa score', (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-4-s1'], neg='No')) +
           cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-4-s1'], neg='No'))) / 2)
    print('acc score: ', accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-4-s1'], neg='No')))
    print("S2 label")
    print('kappa score', (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-4-s2'], neg=False)) +
           cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-4-s2'], neg=False))) / 2)
    print('acc score: ', accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-4-s2'], neg=False)))
    print("S3 label")
    print('kappa score', (
        (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-4-s3-1'], neg='Subjective'))
         + cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-4-s3-2'], neg='Subjective'))) / 2
    ) + (
        (cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-4-s3-1'], neg='Subjective'))
         + cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-4-s3-2'], neg='Subjective'))) / 2
    ) / 2)
    print('acc score', (accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-4-s3-1'], neg='Subjective'))
                        + accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-4-s3-2'], neg='Subjective'))) / 2)
    print("Aggregated label")
    print('gpt4-human kappa score', (cohen_kappa_score(sub_df['label_1'], sub_df['gpt4_label']) +
           cohen_kappa_score(sub_df['label_2'], sub_df['gpt4_label'])) / 2)
    print('inter-human kappa score', cohen_kappa_score(sub_df['label_1'], sub_df['label_2']))
    print('acc score: ', accuracy_score(sub_df['Golden'], sub_df['gpt4_label']))
    print("\n===GPT-3.5===")
    #sub_df = df.loc[(df['gpt-3.5'] < 1) | (df['gpt-3.5'] > 1.5), :]
    sub_df = df
    print("S1 label")
    print('kappa score', (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-3.5-s1'], neg='No')) +
           cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-3.5-s1'], neg='No'))) / 2)
    print('acc score: ', accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-3.5-s1'], neg='No')))
    print("S2 label")
    print('kappa score', (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-3.5-s2'], neg=False)) +
           cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-3.5-s2'], neg=False))) / 2)
    print('acc score: ', accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-3.5-s2'], neg=False)))
    print("S3 label")
    print('kappa score', (
                  (cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-3.5-s3-1'], neg='Subjective'))
                   + cohen_kappa_score(sub_df['label_1'], mapping(sub_df['gpt-3.5-s3-2'], neg='Subjective'))) / 2
          ) + (
                  (cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-3.5-s3-1'], neg='Subjective'))
                   + cohen_kappa_score(sub_df['label_2'], mapping(sub_df['gpt-3.5-s3-2'], neg='Subjective'))) / 2
          ) / 2)
    print('acc score', (accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-3.5-s3-1'], neg='Subjective'))
                        + accuracy_score(sub_df['Golden'], mapping(sub_df['gpt-3.5-s3-2'], neg='Subjective'))) / 2)
    print("Aggregated label")
    print('3.5-human kappa', (cohen_kappa_score(sub_df['label_1'], sub_df['gpt35_label']) +
           cohen_kappa_score(sub_df['label_2'], sub_df['gpt35_label'])) / 2)
    print('human kappa', cohen_kappa_score(sub_df['label_1'], sub_df['label_2']))
    print('acc score: ', accuracy_score(sub_df['Golden'], sub_df['gpt35_label']))
    print('human acc: ', (accuracy_score(sub_df['Golden'], sub_df['label_1'])
                          + accuracy_score(sub_df['Golden'], sub_df['label_2'])) / 2)
    print("\n\n")

    for model_name in ['gpt-3.5', 'gpt-4', 'zephyr', 'llama']:
        golden = df.loc[(df[model_name] == 0) | (df[model_name] == 3), 'Golden'].to_list()
        label_1 = df.loc[(df[model_name] == 0) | (df[model_name] == 3), 'label_1'].to_list()
        label_2 = df.loc[(df[model_name] == 0) | (df[model_name] == 3), 'label_2'].to_list()
        label = df.loc[(df[model_name] == 0) | (df[model_name] == 3), model_name].apply(lambda x: 0 if x == 0 else 1)
        human_kappa = cohen_kappa_score(label_1, label_2)
        ai_acc = accuracy_score(golden, label)
        ai_kappa = (cohen_kappa_score(label, label_2) + cohen_kappa_score(label, label_1)) / 2
        human_acc = (accuracy_score(golden, label_2) + accuracy_score(golden, label_1)) / 2
        print('kappa of inconsistent samples', ai_kappa, human_kappa)
        print('acc of inconsistent samples', ai_acc, human_acc)
        golden = df.loc[(df[model_name] > 0) & (df[model_name] < 3), 'Golden'].to_list()
        label_1 = df.loc[(df[model_name] > 0) & (df[model_name] < 3), 'label_1'].to_list()
        label_2 = df.loc[(df[model_name] > 0) & (df[model_name] < 3), 'label_2'].to_list()
        label = df.loc[(df[model_name] > 0) & (df[model_name] < 3), model_name].apply(lambda x: 0 if x <= 1.5 else 1)
        human_kappa = cohen_kappa_score(label_1, label_2)
        ai_acc = accuracy_score(golden, label)
        ai_kappa = (cohen_kappa_score(label, label_2) + cohen_kappa_score(label, label_1)) / 2
        human_acc = (accuracy_score(golden, label_2) + accuracy_score(golden, label_1)) / 2
        print('kappa of perfect consistent samples', ai_kappa, human_kappa)
        print('acc of perfect consistent samples', ai_acc, human_acc)
        print('\n')


def confusion(num_answer, model, data):
    column_names = ['veri_Answer_' + str(i) for i in range(1, num_answer + 1)]
    if data == 0:
        df = pd.read_excel('data/CoT_self-consistency/policlaim_test_' + model + '_CoT.xlsx')[column_names + ['Golden']]
    elif data == 1:
        df = pd.read_excel('data/CoT_self-consistency/clef2021_test_' + model + '_CoT.xlsx')[column_names + ['Golden']]
    np.random.seed(42)
    rand_labels = np.random.randint(2, size=len(df))
    df['random'] = rand_labels

    def aggregate_func(row):
        answer_list = []
        num = len(column_names)
        for name in column_names:
            if row[name].startswith('Yes'):
                answer_list.append(1)
            else:
                answer_list.append(0)
        answer_sum = sum(answer_list)
        if answer_sum <= num // 2:
            aggregated_answer = 0
            confusion_level = answer_sum
        else:
            aggregated_answer = 1
            confusion_level = num - answer_sum
        return aggregated_answer, confusion_level

    df[['aggregated', 'confusion']] = df.apply(aggregate_func, axis=1, result_type='expand')

    confusion_scores = []
    random_scores = []
    percentage = []
    for i in range(num_answer // 2 + 1):
        agg = df.loc[df['confusion'] == i, 'aggregated']
        golden = df.loc[df['confusion'] == i, 'Golden']
        rand = df.loc[df['confusion'] == i, 'random']
        print('confusion level', i, accuracy_score(golden, agg), 'Random', accuracy_score(rand, agg), "Percentage {:.2f}".format(100 * len(agg) / len(df)))
        confusion_scores.append(accuracy_score(golden, agg))
        random_scores.append(accuracy_score(rand, agg))
        percentage.append(round(100 * len(agg) / len(df), 2))
    # print('majority-voted accuracy: ', confusion_scores)
    # print('random scores: ', random_scores)
    # print('percentage of each consistency level: ', percentage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=int, default=0)
    parser.add_argument("--num_answer", type=int, default=0)
    parser.add_argument("--model", type=str, default='G3')
    args = parser.parse_args()
    if args.data == 0:
        df = pd.read_excel('data/PoliClaim_test/policlaim_gpt_with_human_eval_merged.xlsx')
    else:
        df = pd.read_excel('data/CLEF-2021_test/CLEF2021_human_eval.xlsx')
    if args.num_answer == 0:
        main(df)
    else:
        confusion(args.num_answer, args.model, args.data)