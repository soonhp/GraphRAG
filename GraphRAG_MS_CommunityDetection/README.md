# Implementing Microsoft GraphRAG in Neo4j(GraphRAG)

## 1. DATA LOAD 
### source code : database_generate.py, data_load.py, community_summary.py
   
### Details
   
> 주요 내용은 [LLMGraphTransformer](https://api.python.langchain.com/en/latest/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html) 라이브러리를 활용하여 entity와 relation을 추출하고 이후에 entity 노드 id 혹은 discription의 텍스트 임베딩 값을 neo4j gds라이브러리의 [KNN](https://neo4j.com/docs/graph-data-science/current/algorithms/knn/)과 [WCC(Weakly Connected Components)](https://neo4j.com/docs/graph-data-science/current/algorithms/wcc/)를 활용하여 유사한 엔티티는 병합을 해주고 [Leiden](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=Evergreen&utm_content=EMEA-Search-SEMCE-DSA-None-SEM-SEM-NonABM&utm_term=&utm_adgroup=DSA&gad_source=1&gclid=CjwKCAjwoJa2BhBPEiwA0l0ImNw6uRa44y_xG9LdbkIO9r_NjjE_Mhy_D6nAmR3Tql32YS-FJkVJpBoCw0MQAvD_BwE) 알고리즘으로 커뮤니티를 구성합니다. 
>
   
### How to Run Files
1. database 생성
```
python database_generate.py --DATABASE [name]
```
   [name] 은 예를 들어 graphrag-doc-test 로 본인이 생성하고 싶은 DB 이름을 입력.(따옴표 넣을 필요 없음)
   
   시작하기 앞서 첫 번째로 DB를 생성하는 이유는 database를 설정을 안해주면 default 값인 `neo4j` 로 잡히기 때문에 테스트 단계에서는 생성을 해줘야 한다.

   
   2. data 전처리 후 DB 적재 및 Community Detection
```
python data_load.py --DATABASE [name] --DATA_PATH [File Directory]
```
   `File Directory` 에 데이터(파일) 경로 설정.
   
3. 탐지된 커뮤니티 별 Summary 생성
```
python community_summary.py --level1 [int] --level2 [int] --level3 [int]
```
   [int]에는 어떠한 level 단의 summary를 생성할 지 설정

   
   
   
## 2. Retrieval
### source code : retrieval.py
### Details
> [1. DATA LOAD](#1-data-load)를 통해 구성된 커뮤니티를 활용하여 어떻게 Retrieval을 할 지에 대한 내용이 담겨있습니다. Local과 Global로 나누어서 Retrieval이 진행되고 두 방식의 아키텍쳐가 다르니 직접 들어가서 확인해보시면 될 것 같습니다.
>    
> ms_graphrag_import.ipynb 와 ms_graphrag_retriever.ipynb 를 활용하여 비정형 PDF 예시 문서를 'graphrag-doc-02-ms' 에 적재하였으니 참고바랍니다.
>
### How to Run Files
1. Retrieval & Answer Generation
```
python retrieval.py --index_name [vector_index_name] --question [query]
```
   [vector_index_name] 은 Neo4j를 통해 임베딩 벡터에 대해 인덱싱할 이름 설정

   [query]는 user query를 입력

   참고로 Retrieval & Answer Generation 방식은 아래 그림의 Local retriever로 답변 생성함.
   
### Local retriever
![Alt text](https://miro.medium.com/v2/resize:fit:720/format:webp/1*lInV6WWTDXYEVI1NS3KV9g.png)
   
   Local Search 방법론은 유저 쿼리로부터 엔티티를 인식하여 그 엔티티와 유사한 엔티티가 포함된 1. Source chunk, 2. Community Summary, 3. Entities, 4. Relationship description, 5. Covariate 를 가져와서 질문에 도움이 되지 않는 답변을 필터링하고 각 기준에 따라 점수를 매겨(Ranking) 내림차순으로 토큰 제한에 도달하기 전까지 context를 구성하여 LLM에 보내 최종 답변을 생성합니다.
   
### Global retriever
![Alt text](https://miro.medium.com/v2/resize:fit:720/format:webp/1*mcDNDMTmCqVAUv1SnzTtzA.png)
   
   Global Search 방법론은 우선 모든 Community Summary를 랜덤하게 섞은 후 LLM에게 0~100점까지 점수를 매겨서 도움이 되지 않는다고 판단된 Summary(점수 0)는 필터링합니다. 이 후 생성된 중간 커뮤니티 답변(Rated Intermediate Response)들을 유용성 점수에 따라 내림차순으로 정렬하고, 토큰 제한에 도달할 때까지 가장 유용한 답변들을 선택하여 새로운 context 윈도우에 추가합니다. 이 최종 context를 사용하여 LLM에 보내 유저에게 글로벌 답변을 생성합니다. 이 방식이 위 논문에서 제안한 방법이고 그에 대한 아키텍처입니다.


   Local Search 와 Global Search로 나눈 이유는 미시적 / 거시적 질문에 따라서 취사 선택(Routing)하여 답변을 받을 수 있도록 하기 위함입니다. 이러한 제공된 방법론을 잘 활용하면 어떠한 level 단의 질문에도 잘 답변할 수 있는 GraphRAG를 구축할 수 있을 것입니다.
## Reference
[1. DATA LOAD](#1-data-load) : [Implementing ‘From Local to Global’ GraphRAG with Neo4j and LangChain: Constructing the Graph](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/)
   
[2. Retrieval](#2-retrieval) : [Integrating Microsoft GraphRAG into Neo4j](https://towardsdatascience.com/integrating-microsoft-graphrag-into-neo4j-e0d4fa00714c)
   
Paper : [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/pdf/2404.16130)
   
Microsoft Github Blog : [Microsoft_GraphRAG](https://microsoft.github.io/graphrag/)
