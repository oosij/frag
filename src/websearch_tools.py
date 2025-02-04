import requests 
from requests import get
from bs4 import BeautifulSoup
import json 
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig

TRUNCATE_SCRAPED_TEXT = 5000  

api_key = "google_search_api_key"
cse_id = "google_cse_id"

def websearch_run(search_query):
    search_term = query_to_keyword_by_llm(search_query).replace('"','')
    search_items = search(search_item=search_term, api_key=api_key, cse_id=cse_id, search_depth=10, site_filter=None)
    results = get_search_results(search_items,search_term)
    final_response = websearch_response(search_query, search_term, results)

    return final_response

def search(search_item, api_key, cse_id, search_depth=10, site_filter=None):
    service_url = 'https://www.googleapis.com/customsearch/v1'

    params = {
        'q': search_item,
        'key': api_key,
        'cx': cse_id,
        'num': search_depth
    }

    try:
        response = requests.get(service_url, params=params)
        response.raise_for_status()
        results = json.loads(response.content.decode('utf-8'))
        if 'items' in results:
            if site_filter is not None:
                
                filtered_results = [result for result in results['items'] if site_filter in result['link']]

                if filtered_results:
                    return filtered_results
                else:
                    print(f"No results with {site_filter} found.")
                    return []
            else:
                if 'items' in results:
                    return results['items']
                else:
                    print("No search results found.")
                    return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the search: {e}")
        return []

def retrieve_content(url, max_tokens=TRUNCATE_SCRAPED_TEXT):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            text = soup.get_text(separator=' ', strip=True)
            characters = max_tokens * 4  
            text = text[:characters]
            return text
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return None


def get_search_results(search_items, search_term, character_limit=500):
    results_list = []
    for idx, item in enumerate(search_items, start=1):
        url = item.get('link')
        
        snippet = item.get('snippet', '')
        web_content = retrieve_content(url, TRUNCATE_SCRAPED_TEXT)
        
        if web_content is None:
            print(f"Error: skipped URL: {url}")
        else:
            summary = summarize_content(web_content, search_term, character_limit)
            result_dict = {
                'order': idx,
                'link': url,
                'title': snippet,
                'Summary': summary
            }
            results_list.append(result_dict)
    return results_list

def query_to_keyword_by_llm(search_query):
    messages=[
        {"role": "system", "content": "Provide a google search term based on search query provided below in 3-4 words in korean"},
        {"role": "user", "content": search_query}]

    chatml = chatml_tokenizer_non_chat(messages)
    search_term = inference_llm(chatml, stream = False)
    return search_term


def inference_llm(prompt, stream = False):
    llm_host = "TASK_inferce_api_url"
    url = f"{llm_host}/v2/streamText"
    
    temperature = 0.25
    top_p = 0.9 
    max_tokens = 16384 
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
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

def chatml_tokenizer_non_chat(chat):
    TOKEN_PATH =  './token_files/models--microsoft--phi-4/snapshots/f957856cd926f9d681b14153374d755dd97e45ed/'

    tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH, local_files_only=True)
    chatml = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return chatml

def summarize_content(content, search_term, character_limit=500):
        prompt = (
            f"You are an AI assistant tasked with summarizing content relevant to '{search_term}'. "
            f"Please provide a concise summary in {character_limit} characters or less."
        )
        try:
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}] 
            chatml = chatml_tokenizer_non_chat(messages)
            summary = inference_llm(chatml, stream = False)
            return summary
        except Exception as e:
          
            return None

def websearch_response(search_query, search_term, results):
    final_prompt = (
        f"The user will provide a dictionary of search results in JSON format for search query {search_term} Based on on the search results provided by the user, provide a detailed response to this query: **'{search_query}'**. Make sure to cite all the sources at the end of your answer. The answer must be provided strictly in the same language as the query."
    )

    messages=[
        {"role": "system", "content": final_prompt},
        {"role": "user", "content": json.dumps(results, ensure_ascii=False)}
    ]
    
    chatml = chatml_tokenizer_non_chat(messages)
    return chatml
