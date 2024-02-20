import pandas as pd
import argparse
import random
import pickle as pkl
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
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

PROMPT_PART_1_VERIFIABILITY_CoT_CONTEXT = """Given the <context> of the following <sentence> from a political speech, does it contain any objective information? 

<context>: "...{context}..."
<sentence>: "{sentence}" 

Format your reply as follows:

[Chain of thought]: your step-by-step reasoning about the question
[Answer]: a single word yes or no
"""

PROMPT_PART_1_VERIFIABILITY_GENERAL = """Does the following <sentence> ({origination}) contain any objective information? 

<sentence>: "{sentence}" 

Format your reply as follows:

[Chain of thought]: your step-by-step reasoning about the question
[Answer]: a single word yes or no
"""


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


def _find_answer(string, name="FACT_PART"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            return l[start:end]
    return string


def parse_CoT(all_answers):
    all_CoTs = []
    all_parsed_answers = []
    for answers in all_answers:
        parsed_CoT = []
        parsed_answers = []
        for a in answers:
            try:
                parsed_CoT.append(a.split('[Chain of thought]:')[1].split('[Answer]:')[0].strip())
                parsed_answers.append(a.split('[Answer]:')[1].strip())
            except Exception as e:
                print(a)
                parsed_CoT.append(a)
                parsed_answers.append('No')
        all_CoTs.append(parsed_CoT)
        all_parsed_answers.append(parsed_answers)
    return all_CoTs, all_parsed_answers


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


def verifiability(args, llm, prompt, sentences):
    if args.context == 0 and args.origination != '':
        verifiability_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt.format(sentence=s, origination=args.origination))
            ]
            for s in sentences
        ]
    else:
        contexts = contextualize_sentences(sentences, window_size=args.context)
        verifiability_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt.format(context=c, sentence=s))
            ]
            for c, s in zip(contexts, sentences)
        ]

    verifiability_outputs = asyncio.run(async_api_call(llm, verifiability_prompts, args.num_gen))
    with open(args.output_name + '.pkl', 'wb') as f:
        pkl.dump(verifiability_outputs, f)
    verifiability_CoTs, verifiability_answers = parse_CoT(verifiability_outputs)
    if args.num_gen > 1:
        aggregated_answer, confusion = majority_vote_p1_verifiability(verifiability_answers)
    else:
        aggregated_answer = [l[0] for l in verifiability_answers]
        confusion = None
    df_p1_verifiability = pd.DataFrame({'SENTENCES': sentences})
    for i in range(args.num_gen):
        df_p1_verifiability['veri_CoT_' + str(i + 1)] = [l[i] for l in verifiability_CoTs]
        df_p1_verifiability['veri_Answer_' + str(i + 1)] = [l[i] for l in verifiability_answers]
    df_p1_verifiability['veri_aggregated'] = aggregated_answer
    if confusion is not None:
        df_p1_verifiability['veri_confusion'] = confusion

    # df_p1_verifiability.to_csv(args.output_name + '_ver_p1_' + str(args.num_gen) + '.csv', encoding='utf-8', index=False)
    return df_p1_verifiability


def main(args):
    if args.origination == '':
        P1_VERIFIABILITY = PROMPT_PART_1_VERIFIABILITY_CoT_CONTEXT
    else:
        P1_VERIFIABILITY = PROMPT_PART_1_VERIFIABILITY_GENERAL

    if args.seed > 0:
        random.seed(args.seed)

    if args.file_name.endswith('xlsx'):
        df = pd.read_excel(args.file_name)
    else:
        df = pd.read_csv(args.file_name, encoding='utf-8')
    sentences = df['SENTENCES'].to_list()
    sentences = [s.strip() for s in sentences]
    golden_labels = df['Golden']

    if args.sample > 0:
        sentences = random.sample(sentences, args.sample)

    if args.num_gen > 1:
        temperature = 0.7
    else:
        temperature = 0

    llm = ChatOpenAI(model_name=args.llm_name, temperature=temperature, max_tokens=512)

    df_p1_verifiability = verifiability(args, llm, P1_VERIFIABILITY, sentences)
    df_p1_verifiability['Golden'] = golden_labels
    df_p1_verifiability.to_excel(args.output_name + '_CoT_' + str(args.num_gen) + '.xlsx', index=False)


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--origination", type=str, default="")
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gen", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()
    with get_openai_callback() as cb:
        main(args)
        print(cb)