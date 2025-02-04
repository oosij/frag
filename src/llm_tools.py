import requests
import numpy as np
import pandas as pd
import json
import ast

import re
import os 

import time
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


def chatml_tokenizer_by_check(qta_prompt):
    TOKEN_PATH = "./token_files/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0"
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH, local_files_only=True)

    chat = []
    chat = chat_message_add(chat, qta_prompt)
    chatml = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return chatml


def inference_gemma2(prompt):
    llm_host = "Gemma-infernce_api_url"
    url = f"{llm_host}/v1/generateText"
    
    temperature = 0.25
    top_p = 0.85
    max_tokens = 4096
    
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    answer = response.json()
    answer_output = answer['text'].strip()
    return answer_output


def chatml_tokenizer(qta_prompt):
    TOKEN_PATH =  './token_files/models--microsoft--phi-4/snapshots/f957856cd926f9d681b14153374d755dd97e45ed/'
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH, local_files_only=True)

    chat = []
    chat = chat_message_add(chat, qta_prompt)
    chatml = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return chatml

class MyEmbeddingFunction:
    def __init__(self):
        self.vmodel = SentenceTransformer("dragonkue/BGE-m3-ko")

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]

        embeddings = self.vmodel.encode(input)
        
        return embeddings

def single_llm_answer(instruction, query_doc, query_n):
    chat = []

    prompt_input = query_n_input_cot_load(query_doc, instruction ,'' ,query_n, '')
    answers = inference_llm(prompt_input, False)
    
    return answers , prompt_input

def query_n_input_cot_load(sys_data, instruction, qa_list ,query, reason):
    kor_prompt_template = """You are a clever chatbot who answers questions and provide a rationale of your answers.
Read the Instruction below and provide an answer.

Context:
{doc_}

Instruction: 
{instruction}

{qa_list}

Question: 
{user_query}

Answer only in Korean.
{reason}
##Answer:"""

    prompt_input = kor_prompt_template.format(doc_ =sys_data ,instruction = instruction, qa_list = qa_list, user_query = query, reason = reason)
    return prompt_input

def chat_message_add(chat, prompt_input):
    if len(chat) == 0 :
        chat = [
            { "role": "user", "content": prompt_input},
        ]
    else:
        message = { "role": "user", "content": prompt_input}
        chat.append(message)
    return chat


def cot_llm_answer(instruction, query_doc, query_n, cot_cnt ):
    instruction_cot = """당신은 Context 내용을 참고해서 아래의 형식과 같이 지침에 따라 질의응답 형식을 만들어야 합니다.

만드는 형식의 지침 예는 다음과 같습니다 : 
Question: 질문 1의 내용 
Answer: 질문 1의 답변 

Question: 질문 2의 내용 
..."""
    
    query_cot = "이상의 내용으로 최종 질문에 도달할 수 있도록 질문과 답변을 "+ str(cot_cnt) +"개 만들어주세요."

    reason = """reasoning for that answer. Please use the format of: 
##Reason:"""
    prompt_list = []
    chat = []

    for i in range(2):
        if i == 0 :
            prompt_input = query_n_input_cot_load(query_doc, instruction_cot, '' ,query_cot, '')        
            outputs = inference_llm(prompt_input, False)
            qa_list = answer_cot_preprocess(outputs)
        else:
            chat = []
            prompt_input = query_n_input_cot_load(query_doc, instruction ,qa_list ,query_n, reason)
            answers = inference_llm(prompt_input, False)
        prompt_list.append(prompt_input)
    return answers, qa_list, prompt_list

def answer_cot_preprocess(answers):
    qa_list = []
    answer_query = answers.replace('질의응답','').strip()
    answer_query = answer_query.replace('질문 및 답변','').strip()
    answer_query = answer_query.replace('질문과 답변','').strip()
    answer_query = answer_query.replace('#','').strip()
    qa_list = answer_query.replace('**','').strip()
    return qa_list

def cot_answer_clean(cot_answer):
    shap_a = '##Answer:'
    shap_a_len = cot_answer.find(shap_a)
    cot_answer_ = cot_answer[shap_a_len:]
    final_answer = cot_answer_.replace('##Answer:','').strip()
    if final_answer.find('##Reason:') >=0 :
        reason = final_answer.find('##Reason:')
        final_answer = final_answer[:reason]
        final_answer = final_answer.strip()
    
    return final_answer

def inference_llm(prompt, stream = False):
    llm_host = "TASK_inference_api_url"
    url = f"{llm_host}/v2/streamText"
    
    temperature = 0.25 
    top_p = 0.9
    max_tokens = 8192 
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens * 2
    }

    use_streaming = stream  

    response = requests.post(url, headers=headers, json=data, stream= use_streaming)

    answers = ''

    if use_streaming:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            print(chunk, end="")
            answers += chunk
    else:
        if response.status_code == 200:
            answers = response.text
    return answers.strip()
