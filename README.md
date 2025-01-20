# frag
- only local model & open source (web search : google search api)



** RAG & Web_search & Chat (AI agent Logic) case : 

RAG case : 

![RAG](https://github.com/user-attachments/assets/da135105-987c-49c1-b419-1bfa218ceda0)


Web search case : 

![웹검색](https://github.com/user-attachments/assets/639c854d-76fa-47c5-a997-504b3e6ee020)


Chat case : 

![일상대화](https://github.com/user-attachments/assets/ec9606e7-8113-40de-96fc-1209c80038c4)


-> 현재는 AI agent 로직을 추가하여 통합 및 개선 중 




** RAG sample (old version): 

Question 1 : 
![Q1-_online-video-cutter com_](https://github.com/user-attachments/assets/f2eb3e58-5c49-41d6-852c-828c66d00556)


Question 2 : 
![Q2-_online-video-cutter com_](https://github.com/user-attachments/assets/7ff8999c-e5fd-49de-b82d-41bf03ce612c)


Question 3 : 
![Q3-_online-video-cutter com_](https://github.com/user-attachments/assets/9fde0c19-01cc-4732-a3b3-3d5d4cec99df)


bad case : 
![I1-_online-video-cutter com_](https://github.com/user-attachments/assets/66d3f4a2-b02c-463d-b8c2-1ec75e3d94f9)





raw data : 증권사 리포트 11월 pdf (한달, total 14702 pages) 

![image](https://github.com/user-attachments/assets/fa09e447-a637-4b1d-96ec-640c57ba44a1)




개선점 :
- 각 출처에 링크 삽입 (준비는 됬으나, 현 테스트 단계에서는 생략 -> 현재는 완료)
- 검색 알고리즘 성능 향상 필요 
- latency 및 기타 속도 관련 개선 필요
- 성능 평가 도입 필요 : 구상중인 방법 deepeval 등. (-> 판별기 추가)
- 웹 검색 모듈 추가 (완료)
- ai agent 로직 추가 (진행 중)
- 그외에 필요한 모든 것 등 
