import os
import getpass
from neo4j import GraphDatabase, Result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from answer_generation_chain import answer_chain
from langchain_core.output_parsers import StrOutputParser

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import VectorStoreIndex
from tqdm import tqdm

from parser import parse_args

from typing import Dict, Any

args = parse_args()

# Adjust pandas display settings
pd.set_option(
    "display.max_colwidth", None
)  # Set to None to display the full column width
pd.set_option("display.max_columns", None)

import os
from dotenv import load_dotenv, find_dotenv


# %% Environment setting


class GoRAGRetrieval:
    def __init__(
        self,
        vector_index_name: str,
        top_chunks: int = 3,
        top_communities: int = 3,
        top_outside_rels: int = 10,
        top_inside_rels: int = 10,
        top_entities: int = 10,
    ):
        load_dotenv(find_dotenv())

        # 환경 변수 설정
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.NEO4J_DATABASE = args.DATABASE
        self.index_name = vector_index_name

        # Neo4j 드라이버 생성
        self.driver = GraphDatabase.driver(
            self.NEO4J_URI,
            auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD),
            database=self.NEO4J_DATABASE,
        )

        self.top_chunks = top_chunks
        self.top_communities = top_communities
        self.top_outside_rels = top_outside_rels
        self.top_inside_rels = top_inside_rels
        self.top_entities = top_entities

        self.create_vector_index()
        self.update_community_weights()
        self.lc_vector = self.create_vector()

    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """Cypher 쿼리를 실행하고 DataFrame으로 결과를 반환합니다."""
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return pd.DataFrame([record.data() for record in result])

    def create_vector_index(self):
        """Neo4j에 벡터 인덱스를 생성합니다."""
        cypher = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
        OPTIONS {{indexConfig: {{
         `vector.dimensions`: 1536,
         `vector.similarity_function`: 'cosine'
        }} }}
        """
        self.db_query(cypher)

    def update_community_weights(self):
        """커뮤니티의 가중치를 업데이트합니다."""
        cypher = """
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """
        self.db_query(cypher)

    def create_vector(self) -> Neo4jVector:
        """Neo4jVector 인스턴스를 생성합니다."""
        lc_retrieval_query = f"""
        WITH collect(node) as nodes
        // Entity - Text Unit Mapping
        WITH
        collect {{
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]->(c:Document)
            WITH c, count(distinct n) as freq
            RETURN c.text AS chunkText
            ORDER BY freq DESC
            LIMIT {self.top_chunks}
        }} AS text_mapping,
        // Entity - Report Mapping
        collect {{
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH c, c.rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT {self.top_communities}
        }} AS report_mapping,
        // Outside Relationships 
        collect {{
            UNWIND nodes as n
            MATCH (n)-[r]-(m) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.rank, r.weight DESC 
            LIMIT {self.top_outside_rels}
        }} as outsideRels,
        // Inside Relationships 
        collect {{
            UNWIND nodes as n
            MATCH (n)-[r]-(m) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.rank, r.weight DESC 
            LIMIT {self.top_inside_rels}
        }} as insideRels,
        // Entities_ID
        collect {{
            UNWIND nodes as n
            RETURN n.id AS IDText
        }} as entities_id,
        // Entities_description
        collect {{
            UNWIND nodes as n
            RETURN n.id + " : " + n.description AS descriptionText
        }} as entities_desc
        // We don't have covariates or claims here
        RETURN {{Chunks: text_mapping, Reports: report_mapping, 
               Relationships: outsideRels + insideRels, 
               Entities_ID: entities_id,
               Entities_description : entities_desc}} AS text, 1.0 AS score, {{Entity : entities_id, Community_summary : report_mapping}} AS metadata
        """

        return Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=self.NEO4J_URI,
            username=self.NEO4J_USERNAME,
            password=self.NEO4J_PASSWORD,
            index_name=self.index_name,
            retrieval_query=lc_retrieval_query,
        )

    def setup_retrieval_chain(self) -> Any:
        """검색 체인을 설정합니다."""
        lc_retrieval = self.lc_vector.as_retriever(
            search_kwargs={"k": self.top_entities}
        )
        return create_retrieval_chain(lc_retrieval, answer_chain)

    def ask_question(self, question: str) -> Dict[str, Any]:
        """질문을 하고 답변을 반환합니다."""
        kg_qa = self.setup_retrieval_chain()
        res = kg_qa.invoke({"input": question})
        return res


# 예제 사용법
GoRAG_Gen_Ans = GoRAGRetrieval(vector_index_name=args.index_name)

answer = GoRAG_Gen_Ans.ask_question(question=args.question)
print(f"{answer} \n\n *답변*\n {answer['answer']}")
