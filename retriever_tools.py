import requests
import numpy as np
import pandas as pd
import json
import ast

import re
import os 
import time

from kss import split_sentences  
from konlpy.tag import Kkma
from mecab import MeCab
from rank_bm25 import BM25Okapi
from datetime import datetime, timedelta
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, HasIdCondition, PointStruct

from llm_tools import MyEmbeddingFunction

embedding_function = MyEmbeddingFunction() 

def ensemble_search_query(query_input, all_points, retrievers, weights, options, N=100, context_limit=3, c=60):
    retriever_docs = []
    for retriever in retrievers:
        retriever_docs.append(retriever(query_input, all_points, N, options))  # 리트리버 호출

    combined_results = rrf_rank_fusion(retriever_docs, weights, c)

    sorted_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    sent_result = []
    ids_result = []
    
    for i, rdata in enumerate(sorted_results[:context_limit]):
        stockname, text, combined_score, uuid_ids = rdata["stockname"], rdata["text"], rdata["score"], rdata["id"]
        rwdata = stockname + '\n' + text  
        sent_result.append(rwdata)
        ids_result.append(uuid_ids)

    return sent_result, ids_result


def rrf_rank_fusion(retriever_docs, weights, c):
    rrf_score = defaultdict(float)
    doc_metadata = {}

    for docs, weight in zip(retriever_docs, weights):
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc["id"]
            rrf_score[doc_id] += weight / (rank + c)
        
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = doc

    combined_results = [
        {**doc_metadata[doc_id], "score": score} 
        for doc_id, score in rrf_score.items()
    ]
    return combined_results


def bm25_retriever_func(query_input, all_points, N, options):
    tokenized_corpus, corpus, ids_list = tokenized_by_tokens_corpus_ids(all_points)
    bm25_top_n_ids, bm25_top_n_documents, bm25_scores = bm25_base_cacul(
        query_input, tokenized_corpus, corpus, ids_list, N
    )
    
    id_to_stockname = {point.id: point.payload.get("stockname", "") for point in all_points}
    id_to_text = {point.id: point.payload.get("text", "") for point in all_points}

    return [
        {"id": id_, "summary": summary, "score": score, "stockname": id_to_stockname.get(id_, "") , "text": id_to_text.get(id_, "")}
        for id_, summary, score in zip(bm25_top_n_ids, bm25_top_n_documents, bm25_scores)
    ]
    
def vector_retriever_func(query_input, all_points, N, options):
    client, collection_name = options
    query_embedding = embedding_function(query_input)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0],
        limit=N
    )

    id_to_stockname = {point.id: point.payload.get("stockname", "") for point in all_points}
    id_to_text = {point.id: point.payload.get("text", "") for point in all_points}

    return [
        {"id": result.id, "summary": result.payload.get("summary", ""), "score": result.score, 
         "stockname": id_to_stockname.get(result.id, ""), 
         "text": id_to_text.get(result.id, "")}
        for result in search_result
    ]


def tokenized_by_tokens_corpus_ids(all_points):
    tokenized_corpus = []
    corpus = []
    ids_list = []

    points = all_points

    for i in range(len(points)):
        payloads = points[i].payload
        ids = points[i].id
        ids_list.append(ids)
        corpus.append(payloads['snippet'])
        keyword_words = payloads['keyword']
        stockname = payloads['stockname']
        target_keywords = [stockname]+ keyword_words
        tokenized_corpus.append(target_keywords)
    return tokenized_corpus, corpus, ids_list


def bm25_base_cacul(query, tokenized_corpus,corpus, ids_list, N ):
    tokenized_query = exclude_particles(query).split(' ')
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores  = bm25.get_scores(tokenized_query)
    
    documents  = corpus

    top_n_indices = np.argsort(bm25_scores)[-N:][::-1]  

    top_n_ids = [ids_list[i] for i in top_n_indices]
    top_n_documents = [documents[i] for i in top_n_indices]
    top_n_scores = [bm25_scores[i] for i in top_n_indices]

    return top_n_ids, top_n_documents, top_n_scores


def tokenized_pos_words_extract(sentence):
    tokenized_words = exclude_particles(sentence).split(' ')
    return tokenized_words

def exclude_particles(text):
    mecab = MeCab()
    parsed = mecab.pos(text) 
    result = []
    stop_words = ['은', '는', '가', '이']
    for word, pos in parsed:
        if not pos.startswith('JKS') and pos.startswith('NN') :
            if word not in stop_words:
                result.append(word)

    return ' '.join(result)


def retriever_options(dstart, dend):
    start_date = date_to_timestamp(dstart)
    end_date = date_to_timestamp(dend)
    filter_condition = models.Filter(
        must=[

            models.FieldCondition(
                key="timestamp",
                range=models.Range(
                    gte=start_date,  
                    lte=end_date     
                )
            ),

        ]
    )
    return filter_condition

def retriever_similar_query(query_text, filter_condition):
    query_vector = embedding_function(query_text)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0], 
        query_filter=filter_condition, 
        limit=3,  
        with_payload=True,  
    )
    sent_list = []
    for result in search_result:
        ids, name, text =  result.ids, result.payload['stockname'], result.payload['text']
        rsent = name +'\n'+ text
        sent_list.append([ids,rsent])
        
    return sent_list








