from datetime import datetime, timedelta
from collections import defaultdict

from llm_tools import inference_llm, chatml_tokenizer

def query_transf(select, question):
    prompt_template, in_parameter = template_select(select)['template'], template_select(select)['parameter']
    trans_prompt = prompt_template.format(question = question)
    chat = chatml_tokenizer(trans_prompt)
    trans_answer = inference_llm(chat)

    return trans_answer

def retriever_eval_prompt_template(contexts, query):
    prompt_template = """
다음은 RAG 시스템의 Context와 Question 입니다.
Question의 내용에 맞게 Retriever 결과물을 가져왔는지, 각 Context 내용을 확인하세요.
Question에 대한 답변 정보 또는 Question의 의도에 맞게 Context에 내용이 일부라도 포함되어 있다면 'YES', 포함되어 있지 않다면 'NO'로 답변하세요.
Context에 Question의 답변으로써 활용할 수 잇는 정보가 일부라도 포함되어 있다면, 'YES'로 답변하세요.

# Context :
{context}

# Question : 
{query}
"""
    context = ''
    for c in range(len(contexts)):
        conts = contexts[c]
        context = context + '\n\n'+ conts
    context = context.strip()
    retriever_eval_prompt = prompt_template.format(context= context, query = query)
    return  retriever_eval_prompt
    

def query_to_answer_template(query_text, sent_list ):
    nowdate = datetime.now().strftime('%Y-%m-%d')
    today = datetime.now().strftime("%A")
    prompt_template = """Please answer the question using only the provided information.  
If there is insufficient information based on the question and the given data, do not attempt to generate an answer.  
When mentioning dates, use the current date (Current date) below as a reference, and for additional details, carefully consider the provided information in your response.  
- If the information is insufficient, do not elaborate further beyond the above response.  
- Only generate answers in korean.  

## Information:  
- Current date:  
{nowdate}  
{today}  

{doc_}  

Question: {user_query}
Answer:"""
    doc_ = ''
    
    for i in range(len(sent_list)):
        sent_i = sent_list[i]
        if sent_i == "empty context":
            continue
        doc_ = doc_ + '\n\n' + sent_i
        doc_ = doc_.strip()

    in_prompt = prompt_template.format(user_query = query_text, doc_ = doc_, nowdate = nowdate, today=today )
    return in_prompt

def template_select(select):
    select_list = ['multi', 'hyde', 'meta', 'trans', 'rewrite', 'rewrite2']
    template_dic = {'template':'' ,'parameter':''}
    if select == "multi":
        multi_query_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. 
- only question list generate

Original question: {question}"""
        template_dic['template'] = multi_query_template
        template_dic['parameter'] = ['question']

    if select == "hyde": 
        hyde_template = """Imagine you are an expert writing a detailed explanation on the topic: '{question}' 
Your response should be comprehensive and include all key points that would be found in the top search result.
- only generate Korean.
- certainly, only generate three sentence response.

Passage:"""
        template_dic['template'] = hyde_template
        template_dic['parameter'] = ['question']

    if select == "hyde2": 
        hyde2_template = """Please write a passage in Korean to answer the question in detail as if the information is definitely available. 
Avoid responses that suggest uncertainty or indicate missing information (e.g., "not available" or "not yet").
- only generate Korean.
Question: {question}

Passage:"""
        template_dic['template'] = hyde2_template
        template_dic['parameter'] = ['question']

    if select == "hyde3": 
        hyde2_template = """질문에 대한 답변을 상세히 작성하되, 정보를 확실히 제공할 수 있는 것처럼 작성해 주십시오.
불확실성을 암시하거나 정보가 부족하다는 내용(예: '이용할 수 없음', '아직 준비되지 않음')을 포함하지 않도록 해 주십시오.
- 단, 너무 길어지지 않도록 3줄로 제한해주세요.
- 표 내용은 문장 단위로 3줄로 구성하세요. 표로 만드시면 안됩니다.
Question: {question}

Passage:"""
        template_dic['template'] = hyde2_template
        template_dic['parameter'] = ['question']

    if select =="meta" :
        meta_template = """Follow the guidelines below to format the output as shown in the example. 
The guidelines are as follows:
- Identify any dates mentioned in the question (expressed as individual dates or date ranges).
- For the date, measure the range based on the current date. 
- Extract any keywords relevant to potential stock issues from the following question.
- If there is none, do not force the output with missing content.
- Format them as shown in the output example below: 
{{
  "date_range": "YYYY-MM-DD to YYYY-MM-DD",
  "issue_keywords": ["keyword1", "keyword2", "keyword3"]
}}

current date:
{nowdate}
{today}

question:
{question}"""
        template_dic['template'] = meta_template
        template_dic['parameter'] = ['nowdate', 'today','question']
        
    if select == "trans" :
        follow_up_template = """Please transform the content of the question into a question format based on the information in the given metadata.
Only korean generate text. Do not forcefully add information that does not exist.
meta data:
{metadata}

question:{question}

generate:"""
        template_dic['template'] = follow_up_template
        template_dic['parameter'] = ['metadata','question']


    if select == 'rewrite':
        rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
Only korean generate text. Do not forcefully add information that does not exist.
Original query: {question}

Rewritten query:"""
        template_dic['template'] = rewrite_template
        template_dic['parameter'] = ['question']


    if select == 'rewrite2':
        rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query and metadata, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
Only korean generate text. Do not forcefully add information that does not exist.
meta data:
{metadata}

Original query: {question}

Rewritten query:"""
        template_dic['template'] = rewrite_template
        template_dic['parameter'] = ['metadata','question']

    return template_dic

def summmary_for_tabula(title_, table_):
    if title_ is None :
        title_ = ''
  
    doc_ = title_ +'\n'+ table_
    doc = doc_.strip()

    summary_prompt_template = """Please restructure the table data into concise sentences and paragraphs for use in a RAG system. 
Keep the content brief and clear while maintaining the essential information from the table.
If there is a title for the table, start with the table's title and describe its contents; 
if there is no title, represent the content based on the information in the table : 
Only answer in Korean.
The target table data is as follows:
{doc}

answer:"""

    summary_prompt = summary_prompt_template.format(doc = doc)
    summary_output = inference_llm(summary_prompt, False)
    return summary_output

def date_range_time(meta_data):
    date_range = meta_data['date_range']
    start_date_str, end_date_str = date_range.split(" to ")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return start_date, end_date

def extract_json_data(text):
    pattern = r'\{[^{}]*\}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()
    return None