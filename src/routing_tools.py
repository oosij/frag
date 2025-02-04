def create_routing_tools():
    routing_tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_document_tools",
                "description": (
                    "증권가 리포트로 구성된 벡터 DB를 기반으로 유사도 기반 검색 후, 질문과 유사한 단락을 가져옵니다. "
                    "주식 종목들의 실적, 추이, 전망, 표, 투자의견 등의 정보를 담고 있습니다.",
                    "주식 증권사 리포트 관련에 관련된 내용의 경우 우선적으로 해당됩니다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "질문 내용. 예: 이마트 실적 추이를 표로 그려주고 설명해줘."
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search_tools",
                "description": (
                    "구글 API 기반으로 웹 검색을 활용해서 검색 결과인 각 웹 문서의 요약문을 가져옵니다. "
                    "웹을 검색해서 답변이 필요한 경우, 우선적으로 해당됩니다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "질문 내용. 예: 현재 삼성전자의 주가는?"
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "chat_history_tools",
                "description": (
                    "일상적인 대화에 적합합니다. 단, 최신 데이터를 반영하지 못하니, 일상적인 대화 또는 "
                    "상식에 의거한 답변이 필요시 선택하세요. 시간을 요구하는 질문에는 답변을 피하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "질문 내용. 예: 오늘 기분은 어때?"
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]
    return routing_tools