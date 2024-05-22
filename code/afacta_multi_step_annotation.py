import pandas as pd
import argparse
import random
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import json
import asyncio
import numpy as np

os.environ["OPENAI_API_KEY"] = "sk-xxx"

SYSTEM_PROMPT = """You are an AI assistant who helps fact-checkers to identify fact-like information in statements.
"""

PROMPT_PART_2_0905 = """Statements in political speech are usually based on facts to draw reasonable conclusions.

Categories of fact:
C1. Mentioning somebody (including the speaker) did or is doing something specific and objective.
C2. Quoting quantities, statistics, and data.
C3. Claiming a correlation or causation.
C4. Assertion of existing laws or rules of operation.
C5. Pledging a specific future plan or making specific predictions about future.

Please first analyze the objective and subjective information that the following <statement> (from a political speech) covers.
Then extract the fact that the <statement> is based on.
Then carefully reason about if the extracted fact is objectively verifiable. 
Finally answer if the fact falls into the above categories (C1 to C5) or not (C0).

Context for <statement> to help you understand it better: "{context}"
<statement>: "{sentence}"

Format your answer in JSON with the following keys in order: 
{{
    "ANALYSIS": "What are the objective and subjective information that <statement> covers?",
    "FACT_PART": "The extracted fact.",
    "VERIFIABLE_REASON": "Detailed reason about the extracted fact's verifiability. Note that a fact lacks important details or can be interpreted differently is not objectively verifiable. Future plans/pledge (C5) that are specific and clear can be verifiable. Citing others' words is verifiable and falls into C1. ",
    "VERIFIABILITY": "A boolean value indicates the verifiability.",
    "CATEGORY": "C1 to C5, or C0."
}}
"""

PROMPT_PART_1_VERIFIABILITY = """Given the <context> of the following <sentence> from a political speech, does it contain any objective information? 

<context>: "...{context}..."
<sentence>: "{sentence}" 

Answer with Yes or No only.
"""

PROMPT_OBJECTIVE = """Concisely argue that the following <sentence> from a political speech does contain some objective information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"
"""


PROMPT_SUBJECTIVE = """Concisely argue that the following <sentence> from a political speech does not contain any objective information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"
"""

JUDGE_PROMPT = """Two AI assistants are debating about whether the following <sentence> (from a political speech) contains any objectively verifiable information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"

Assistant A's View: "{assistant_a}"

Assistant B's View: "{assistant_b}"

Based on the above, does <sentence> contain any objectively verifiable information? Which perspective do you align with more closely? 
Please reply with "Lean towards A", or "Lean towards B" only."""


def judge_vote(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        lean2a_count = 0
        lean2b_count = 0
        for a in answer_list:
            if "lean towards a" in a.lower():
                lean2a_count += 1
            else:
                lean2b_count += 1
        if lean2a_count > lean2b_count:
            final_answer.append("Lean towards A")
            confusion.append(lean2b_count)
        else:
            final_answer.append("Lean towards B")
            confusion.append(lean2a_count)
    return final_answer, confusion


def majority_vote_p1_opinion(answer_lists):
    final_answer = []
    confusion = []
    total_ans_num = len(answer_lists[0])
    for answer_list in answer_lists:
        opinion_count = 0
        fact_count = 0
        mix_count = 0
        for a in answer_list:
            if "Opinion with fact" in a:
                mix_count += 1
            elif "Fact" in a:
                fact_count += 1
            else:
                opinion_count += 1
        candidates = ['Opinion with fact', 'Fact', 'Opinion']
        max_num = np.max([mix_count, fact_count, opinion_count])
        if max_num > total_ans_num // 3 + 1:
            final_answer.append(candidates[np.argmax([mix_count, fact_count, opinion_count])])
        else:
            final_answer.append('Opinion with fact')
        confusion.append(total_ans_num - max_num)
    return final_answer, confusion


def majority_vote_p1_verifiability(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        yes_count = 0
        no_count = 0
        for a in answer_list:
            if "yes" in a.lower():
                yes_count += 1
            else:
                no_count += 1
        if yes_count > no_count:
            final_answer.append("Yes")
            confusion.append(no_count)
        else:
            final_answer.append("No")
            confusion.append(yes_count)
    return final_answer, confusion


def majority_vote_p2(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        true_count = 0
        false_count = 0
        for a in answer_list:
            if "true" in str(a).lower():
                true_count += 1
            else:
                false_count += 1
        if true_count > false_count:
            final_answer.append("TRUE")
            confusion.append(false_count)
        else:
            final_answer.append("FALSE")
            confusion.append(true_count)
    return final_answer, confusion


def _find_answer(string, name="FACT_PART"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            return l[start:end]
    return string


def parse_part2(all_answers, keys):
    return_lists = {k: [] for k in keys}
    for answers in all_answers:
        lists = {k: [] for k in keys}
        for a in answers:
            try:
                result_dict = json.loads(a)
            except Exception as e:
                result_dict = {
                    k: _find_answer(a, name=k) for k in keys
                }
            for k in keys:
                lists[k].append(result_dict[k])
        for k in keys:
            return_lists[k].append(lists[k])
    return return_lists


def batchify_list(input_list, batch_size):
    batches = []
    for i in range(0, len(input_list), batch_size):
        batches.append(input_list[i:i+batch_size])
    return batches


def contextualize_sentences(sentences, window_size=1):
    contexts = []
    for i, sent in enumerate(sentences):
        context = ""
        for j in range(- window_size, 1 + window_size):
            if 0 <= i + j < len(sentences):
                context += sentences[i + j] + ' '
        contexts.append(context)
    return contexts


async def async_api_call(llm, messages, gen_num, batch_size=10):
    batches = batchify_list(messages, batch_size)
    all_outputs = []
    for b in batches:
        await asyncio.sleep(0.1)
        outputs = await llm.agenerate(b, n=gen_num)
        output_texts = [[g[i].text for i in range(gen_num)] for g in outputs.generations]
        all_outputs.extend(output_texts)
    return all_outputs


def lean_to_answer(answer, first):
    if first == 'objective':
        if "lean towards a" in answer.lower():
            return "Objective"
        elif "lean towards b" in answer.lower():
            return "Subjective"
        else:
            return "Not defined: " + answer
    else:
        if "lean towards b" in answer.lower():
            return "Objective"
        elif "lean towards a" in answer.lower():
            return "Subjective"
        else:
            return "Not defined: " + answer


def compute_likelihood(to_label, golden):
    if to_label.endswith('xlsx'):
        df_to_eval = pd.read_excel(to_label)
    else:
        df_to_eval = pd.read_csv(to_label, encoding='utf-8')
    if golden.endswith('xlsx'):
        df_golden = pd.read_excel(golden)
    else:
        df_golden = pd.read_csv(golden)

    p1 = df_to_eval['veri_aggregated'].apply(lambda x: 1 if "yes" in x.lower() else 0).values
    p2 = df_to_eval['p2_aggregated'].apply(lambda x: 1 if x else 0).values
    p3_1 = df_to_eval['ob_aggregated'].apply(lambda x: 1 if "objective" in x.lower() else 0).values
    p3_2 = df_to_eval['sub_aggregated'].apply(lambda x: 1 if "objective" in x.lower() else 0).values

    df_to_eval['likely'] = p1 + p2 + p3_1 + p3_2
    df_to_eval['likely_2'] = p1 + p2 + ((p3_1 + p3_2) > 0).astype(int)
    df_to_eval['likely_1'] = p1 + p2 + 0.5 * p3_1 + 0.5 * p3_2
    df_to_eval['GOLD'] = df_golden['VERIFIABILITY_GOLDEN']

    df_to_eval.to_excel(to_label.replace('.csv', '.xlsx'), index=False)


def debate(args, llm, sentences):
    if args.load_debate == "":
        objective_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=PROMPT_OBJECTIVE.format(origination=args.origination, sentence=s))
            ]
            for s in sentences
        ]
        objective_outputs = asyncio.run(async_api_call(llm, objective_prompts, 1))
        objective_outputs = [o[0].strip() for o in objective_outputs]
        subjective_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=PROMPT_SUBJECTIVE.format(origination=args.origination, sentence=s))
            ]
            for s in sentences
        ]
        subjective_outputs = asyncio.run(async_api_call(llm, subjective_prompts, 1))
        subjective_outputs = [o[0].strip() for o in subjective_outputs]
        df_debate = pd.DataFrame({"SENTENCES": sentences, 'subjectivity': subjective_outputs, 'objectivity': objective_outputs})
        df_debate.to_csv(args.output_name + '_debate.csv', index=False, encoding='utf-8')
    else:
        df_debate = pd.read_csv(args.load_debate + '_debate.csv', encoding='utf-8')
        subjective_outputs = [l.strip() for l in df_debate['subjectivity'].to_list()]
        objective_outputs = [l.strip() for l in df_debate['objectivity'].to_list()]

    time.sleep(args.sleep)
    df_debate_results = df_debate

    judge_prompt = JUDGE_PROMPT
    objective_first_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=judge_prompt.format(origination=args.origination, sentence=s, assistant_a=ob, assistant_b=sub))
            ]
            for s, ob, sub in zip(sentences, objective_outputs, subjective_outputs)
        ]

    objective_first_outputs = asyncio.run(async_api_call(llm, objective_first_prompts, args.num_gen))

    ob_aggregated_answer = [o[0] for o in objective_first_outputs]
    ob_verifiable_answer = [lean_to_answer(a, first='objective') for a in ob_aggregated_answer]
    df_debate_results['ob_aggregated'] = ob_verifiable_answer

    time.sleep(args.sleep)

    subjective_first_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt.format(origination=args.origination, sentence=s, assistant_a=sub, assistant_b=ob))
        ]
        for s, ob, sub in zip(sentences, objective_outputs, subjective_outputs)
    ]

    subjective_first_outputs = asyncio.run(async_api_call(llm, subjective_first_prompts, args.num_gen))

    sub_aggregated_answer = [o[0] for o in subjective_first_outputs]
    sub_verifiable_answer = [lean_to_answer(a, first='subjective') for a in
                            sub_aggregated_answer]
    df_debate_results['sub_aggregated'] = sub_verifiable_answer

    df_debate_results.to_csv(args.output_name + '_p3_' + str(args.num_gen) + '.csv', index=False, encoding='utf-8')
    return df_debate_results


def opinion(args, llm, prompt, sentences):
    fact_opinion_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(origination=args.origination, sentence=s))
        ]
        for s in sentences
    ]

    opinion_outputs = asyncio.run(async_api_call(llm, fact_opinion_prompts, args.num_gen))

    opinion_answers = [o[0] for o in opinion_outputs]
    df_p1_opinion = pd.DataFrame({'SENTENCES': sentences, 'op_aggregated': opinion_answers})

    df_p1_opinion.to_csv(args.output_name + '_opinion_p1_' + str(args.num_gen) + '.csv', encoding='utf-8', index=False)
    return df_p1_opinion


def verifiability(args, llm, prompt, sentences):
    verifiability_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(origination=args.origination, sentence=s))
        ]
        for s in sentences
    ]

    verifiability_outputs = asyncio.run(async_api_call(llm, verifiability_prompts, args.num_gen))

    verifiability_answers = [o[0] for o in verifiability_outputs]
    df_p1_verifiability = pd.DataFrame({'SENTENCES': sentences, 'veri_aggregated': verifiability_answers})

    df_p1_verifiability.to_csv(args.output_name + '_ver_p1_' + str(args.num_gen) + '.csv', encoding='utf-8', index=False)
    return df_p1_verifiability


def part_2(args, llm, prompt, p2_keys, verifiable_key, sentences):
    part2_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(origination=args.origination, sentence=s))
        ]
        for s in sentences
    ]

    part2_outputs = asyncio.run(async_api_call(llm, part2_prompts, args.num_gen))
    answer_lists = parse_part2(part2_outputs, keys=p2_keys)
    if args.num_gen > 1:
        aggregated_answer, confusion = majority_vote_p2(answer_lists[verifiable_key])
    else:
        aggregated_answer = [l[0] for l in answer_lists[verifiable_key]]
        confusion = None
    df_p2 = pd.DataFrame({'SENTENCES': sentences})
    for i in range(args.num_gen):
        for k in p2_keys:
            df_p2[k + str(i + 1)] = [l[i] for l in answer_lists[k]]
    df_p2['p2_aggregated'] = aggregated_answer
    if confusion is not None:
        df_p2['p2_confusion'] = confusion

    df_p2.to_csv(args.output_name + '_p2_' + str(args.num_gen) + '.csv', encoding='utf-8', index=False)
    return df_p2


def main(args):
    P1_VERIFIABILITY = PROMPT_PART_1_VERIFIABILITY
    PART_2_PROMPT = PROMPT_PART_2_0905

    PART2_KEYS = ["ANALYSIS", "FACT_PART", "VERIFIABLE_REASON", "VERIFIABILITY", "CATEGORY"]
    verifiable_key = "VERIFIABILITY"

    if args.seed > 0:
        random.seed(args.seed)

    if args.file_name.endswith('xlsx'):
        df = pd.read_excel(args.file_name)
    else:
        df = pd.read_csv(args.file_name, encoding='utf-8')
    sentences = df['SENTENCES'].to_list()
    sentences = [s.strip() for s in sentences]

    if args.sample > 0:
        sentences = random.sample(sentences, args.sample)

    if args.num_gen > 1:
        temperature = 0.7
    else:
        temperature = 0

    llm = ChatOpenAI(model_name=args.llm_name, temperature=temperature, max_tokens=512)

    if not args.skip_p1:
        # Part 1 verifiability
        df_p1_verifiability = verifiability(args, llm, P1_VERIFIABILITY, sentences)
        time.sleep(args.sleep)
    else:
        df_p1_verifiability = pd.read_csv(args.load_p1 + '_ver_p1_' + str(args.num_gen) + '.csv', encoding='utf-8')

    # Part 2 annotation
    if not args.skip_p2:
        df_p2 = part_2(args, llm, PART_2_PROMPT, PART2_KEYS, verifiable_key, sentences)
    else:
        df_p2 = pd.read_csv(args.load_p2 + '_p2_' + str(args.num_gen) + '.csv', encoding='utf-8')

    # Part 3 debate annotation
    if not args.skip_p3:
        df_p3 = debate(args, llm, sentences)
    else:
        df_p3 = pd.read_csv(args.load_p3 + '_p3_' + str(args.num_gen) + '.csv', encoding='utf-8')

    df_merged = pd.merge(df_p1_verifiability, df_p2, how='left', on='SENTENCES')
    df_merged = pd.merge(df_merged, df_p3, how='left', on='SENTENCES')
    df_merged.to_csv(args.output_name + '_' + str(args.num_gen) + '.csv', encoding='utf-8', index=False)

    # compute_likelihood(args.output_name + '_' + str(args.num_gen) + '.csv', args.file_name)


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument("--origination", type=str, default="from a political debate")
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--load_debate", type=str, default="")
    parser.add_argument("--load_p1", type=str, default="")
    parser.add_argument("--load_p2", type=str, default="")
    parser.add_argument("--load_p3", type=str, default="")
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--skip_p1", action="store_true", default=False)
    parser.add_argument("--skip_p2", action="store_true", default=False)
    parser.add_argument("--skip_p3", action="store_true", default=False)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gen", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()

    main(args)
