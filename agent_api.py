import time
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, HasIdCondition, PointStruct
from sentence_transformers import SentenceTransformer

from llm_tools import inference_llm, chatml_tokenizer
from eval_tools import retriever_eval_by_judge, refer_output_dic

from vectordb_tools import get_all_points
from prompt_template_tools import query_transf, query_to_answer_template
from retriever_tools import bm25_retriever_func, vector_retriever_func, ensemble_search_query

from websearch_tools import websearch_run

from routing_tools import create_routing_tools
from agent_tools import route_prompt_to_llm


start_time = time.time()

# Embedding model 
class MyEmbeddingFunction:
    def __init__(self):
        self.vmodel = SentenceTransformer("dragonkue/BGE-m3-ko")

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]

        embeddings = self.vmodel.encode(input)
        
        return embeddings

def rag_document_tools(query_text):
    selects = ['multi', 'hyde',  'hyde2', 'hyde3','rewrite'] 
    select = 'hyde3'

    weights = [0.5, 0.5] 
    chat = []

    n = 1000
    context_limits = 3

    query_tf = query_transf(select, query_text)
    retrievers = [bm25_retriever_func, vector_retriever_func]

    # RAG based search 
    contexts, uuids = ensemble_search_query(query_tf, all_points, retrievers, weights, options, n, context_limit=3, c=60)
    reval_response = retriever_eval_by_judge(contexts, query_text)
  
    if reval_response.find('YES') >= 0 :
        qta_prompt = query_to_answer_template(query_text, contexts)
        chat = chatml_tokenizer(qta_prompt) 
        result_data = refer_output_dic(options, uuids, chat)
    else:
        print(f'The selected function is **{function_name}**, and query is **{query_text}**.')
        chat = websearch_run(query_text)
        result_data =  {'token_msg': chat}

    return result_data

def web_search_tools(query_text):
    chat = websearch_run(query_text)
    result_data =  {'token_msg': chat}
    return result_data

def chat_history_tools(query_text):
    query_text = system_message(query_text)
    chat = chatml_tokenizer(query_text)
    result_data =  {'token_msg': chat}
    return result_data

def system_message(query):
    sys_msg = """<SYS>당신은 금융 투자 증권 지식에 해박한 친절한 AI, ThinkAI입니다. instruction을 기반으로 질문에 성실히 답변하세요."""
    inst_msg = """
    \n### instruction
    - 마지막 답변이 끝날때 공백이 있으면 안됩니다.
    - 한자(chinese character)를 절대로 쓰지마세요.
    - 임의의 표시가 아닌 구체적인 표시를 해야만 합니다.
    - 대화를 유도하도록 답변 끝에 연관된 질문을 하세요. 자연스럽게 하셔야 합니다. 
    - "이제까지의 대화 기록: "으로 요약한 내용이 존재한다면, 해당 대화 기록을 참고해서 이어지듯이 자연스럽게 답변하세요. 

    \n
    ## 질문:{query}"""

    return sys_msg + inst_msg.format(query = query)


# 함수 이름과 실제 함수 객체 매핑
function_map = {
    "chat_history_tools": chat_history_tools,
    "rag_document_tools": rag_document_tools,
    "web_search_tools": web_search_tools
}

qdr_db_path = "./qdb"
collection_name = "frdb"

client = QdrantClient(path= qdr_db_path)  # Persists changes to disk
embedding_function = MyEmbeddingFunction() # (n, 1024)
all_points = get_all_points(client, collection_name)
options = [client, collection_name]

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Vector DB Load End => Elapsed time: {elapsed_time:.2f} seconds")

app = FastAPI()

class QueryRequest(BaseModel):
    query_text: str


@app.post("/agent")
def agent_api(req: QueryRequest):
    query_text = req.query_text
    routing_tools_collection = create_routing_tools()
    route_response = route_prompt_to_llm(query_text, routing_tools_collection)
    function_name = route_response["function"]  # 함수 이름 가져오기
    print(f'The selected function is **{function_name}**, and query is **{query_text}**.')

    if function_name in function_map:
        result_data = function_map[function_name](query_text)  # 매핑된 함수 호출

    return result_data

if __name__ == "__main__":
    import uvicorn
    # 로컬 실행 시 uvicorn 으로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=6000, reload=True)
