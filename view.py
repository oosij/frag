import streamlit as st
import requests
import json

from llm_func import inference_llm_steam
from websearch_gpt import websearch_run

# Streamlit app setup
st.set_page_config(page_title="AI Search Engine", layout="centered")
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stTextInput {border: 1px solid #ccc; border-radius: 4px;}
    .search-container {display: flex; justify-content: center; align-items: center; margin-top: 50px;}
    .search-box {width: 70%;}
    .search-button {margin-left: 10px;}
    .result-container {background-color: #ffffff; border: 1px solid #ccc; border-radius: 10px; padding: 20px; margin-top: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    .result-icon {font-size: 1em; margin-right: 5px; color: #4CAF50;}
    .result-header {display: flex; align-items: center; font-size: 1em; font-weight: bold; color: #333;}
    .result-content {font-size: 1.1em; color: #555; margin-top: 10px; white-space: pre-wrap; line-height: 1.5; background-color: #f9f9f9; padding: 10px; border-radius: 5px;}
    .reference-title {font-weight: bold; color: #444; margin-top: 15px;}
    .reference-content {font-size: 0.9em; color: #666;}
    .ai-response {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .divider {border-top: 1px solid #ccc; margin: 15px 0;}
</style>
""", unsafe_allow_html=True)

# Centralized search engine style UI
st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <h1 style="font-size: 2em;">🔍 AI Search Engine</h1>
    <p style="font-size: 1.2em; color: #666;">AI Agent Prototype, Tool used : RAG / Web Search / Casual QA</p>
</div>
""", unsafe_allow_html=True)


def stream_text_in_chunks(text, chunk_size=3):
    """텍스트를 작은 청크로 나누어 반환하는 함수"""
    chars = list(text)
    chunks = []
    current_chunk = []

    for char in chars:
        current_chunk.append(char)
        if len(current_chunk) >= chunk_size:
            chunks.append(''.join(current_chunk))
            current_chunk = []

    if current_chunk:  # 마지막 청크 처리
        chunks.append(''.join(current_chunk))

    return chunks


# Search box and button
col1, col2 = st.columns([8, 1])
with col1:
    query_text = st.text_input("", placeholder="Type your query here.", label_visibility="collapsed", key="query_input")
with col2:
    search_clicked = st.button("search", key="search_button")

# Handle search input
if search_clicked or (query_text and st.session_state.get("query_input", None)):
    if query_text.strip() == "":
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("🤔 AI 생각 중..."):
            # Prepare request payload and headers
            data = {"query_text": query_text}
            headers = {"Content-Type": "application/json"}

            try:
                # Send request to server
                response = requests.post("http://61.33.245.139:6000/agent", headers=headers, json=data)

                if response.status_code == 200:
                    full_response = response.json()
                    prompt = full_response['token_msg']

                    llm_host = "http://121.254.150.96:5000"
                    url = f"{llm_host}/v2/streamText"

                    temperature = 0.25
                    top_p = 0.9
                    max_tokens = 16384

                    data = {
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens
                    }

                    # Create placeholder for streaming response
                    content_placeholder = st.empty()

                    # Send request to LLM server
                    llm_response = requests.post(url, json=data, stream=True)

                    if llm_response.status_code == 200:
                        full_response_text = ""
                        current_line = ""

                        for line in llm_response.iter_lines(decode_unicode=True):
                            if line:
                                try:
                                    # 현재 라인을 작은 청크로 나누기
                                    chunks = stream_text_in_chunks(line, chunk_size=3)

                                    # 각 청크를 순차적으로 표시
                                    for chunk in chunks:
                                        full_response_text += chunk

                                        # 응답 업데이트
                                        content_placeholder.markdown(f"""
                                        <div class="result-container ai-response">
                                            <div class="result-header">
                                                <span class="result-icon">🤖</span> AI Response
                                            </div>
                                            <div class="result-content">{full_response_text}</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        # 짧은 대기 시간 추가 (필요한 경우)
                                        import time

                                        time.sleep(0.05)  # 50ms 대기

                                    # 라인 끝에 줄바꿈 추가
                                    full_response_text += "\n"

                                except Exception as e:
                                    st.error(f"Error processing response: {e}")
                                    continue

                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                        # Display references if available
                        refer_count = len(full_response) - 1
                        for i in range(1, refer_count + 1):
                            reference_key = f'reference_{i}'
                            reference = full_response.get(reference_key, None)
                            if reference:
                                ref_title = reference[0]
                                ref_content = reference[1]
                                with st.expander(f"[출처 {i}] {ref_title}"):
                                    st.markdown(ref_content)
                    else:
                        st.error(f"LLM server error: {llm_response.status_code}")
                else:
                    st.error(f"Server error: {response.status_code}")

            except Exception as e:
                st.error(f"Failed to connect to the server. Error: {e}")
