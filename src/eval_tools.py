import requests
import numpy as np
import pandas as pd
import json
import ast

import re
import os 
import time

from prompt_template_tools import retriever_eval_prompt_template
from llm_tools import chatml_tokenizer_by_check, inference_gemma2

def check_infernece(prompt):
    qta_prompt = chatml_tokenizer_by_check(prompt)
    check_response = inference_gemma2(qta_prompt)
    return check_response

def retriever_eval_by_judge(contexts, query):
    reval_prompt = retriever_eval_prompt_template(contexts, query)
    reval_response = check_infernece(reval_prompt) 
    reval_response = reval_response.upper()
    return reval_response

def refer_output_dic(options, uuids, chat):
    refer_list = reference_by_uuid(options,  uuids)

    result_data =  {'token_msg': chat}

    for r in range(len(refer_list)):
        sname, sdate, spath = refer_list[r][0], refer_list[r][1], refer_list[r][2]
        stext = refer_list[r][3]
        org_code_str = spath.split('_')[2]
        fr_org_name = finreport_extract_org_name(org_code_str)
        write_refer = f"{sdate} 일자의 {sname} 종목의 {fr_org_name} 리포트 파일"
        
        w_refer_n = f'reference_{r+1}'
        result_data[w_refer_n] = [write_refer, stext]
    return result_data

def reference_by_uuid(options,  uuids):
    client, collection_name = options
    retrieved_points = client.retrieve(
        collection_name = collection_name,
        ids = uuids
    )

    refer_datas = []

    for i in range(len(retrieved_points)):
        stock_name = retrieved_points[i].payload['stockname']
        current_date = retrieved_points[i].payload['datetime']
        reference_file = retrieved_points[i].payload['source'].split('/')[-1]
        reference_text = retrieved_points[i].payload['text']

        refer_data = [stock_name, current_date, reference_file, reference_text]
        refer_datas.append(refer_data)

    return refer_datas

def finreport_extract_org_name(organ_code):
    finreport_name_path = './data/finreports/fin_report_name.xlsx' # without DB 
    fn_df = pd.read_excel(finreport_name_path, dtype=str)

    fn_org_data = fn_df[fn_df['organ_code'] == organ_code]
    fn_name =  fn_org_data['organ_name'].iloc[0]
    return fn_name