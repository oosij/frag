import json
import re

from llm_tools import chatml_tokenizer, inference_llm

def route_prompt_to_llm(question, routing_dic):
    prompt_template = """
당신은 라우팅 식별자입니다. 당신의 역할은 다음과 같습니다 : 
당신은 question의 내용에 가장 적합하고, 올바른 답을 할 수 있는 아래의 function을 description의 값을 토대로 알맞은 function와 parameters 출력해야합니다.
question의 내용은 그대로 해당하는 parameters의 값에 있는 그대로 취급하세요.
json 형태의 데이터를 참고해서 질문에 알맞은 function, parameters json 형태로 출력하세요. 다른 부가 설명은 안하셔야합니다.

### 라우팅 판별 대상 데이터 :
{routing_dic}

### question: 
{question}
    """
    route_prompt = prompt_template.format(question=question, routing_dic=routing_dic)
    
    route_prompt_chatml = chatml_tokenizer(route_prompt)
    route_response = inference_llm(route_prompt_chatml, stream = False)

    json_string = re.search(r'```json\n(.*?)\n```', route_response, re.DOTALL).group(1)
    json_dict = json.loads(json_string)

    return json_dict



