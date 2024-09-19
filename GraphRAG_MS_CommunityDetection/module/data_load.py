import getpass
import os
import fitz
import logging
import tiktoken
import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import List

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from graphdatascience import GraphDataScience

from typing import List, Optional

from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from parser import parse_args


# args = parse_args()
# load_dotenv(find_dotenv())


class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


class GraphProcessingPipeline:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        load_dotenv(find_dotenv())
        self.args = parse_args()
        self.set_environment()
        self.graph = Neo4jGraph()
        self.doctext_dict = {}
        self.df = pd.DataFrame()

    def set_environment(self):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
        os.environ["NEO4J_DATABASE"] = self.args.DATABASE
        os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
        os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

    def load_data(self):
        logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")
        logging.info("START: Data load")
        data_path = self.args.DATA_PATH
        print("DATA_PATH : ", data_path)
        dataFileList = os.listdir(data_path)

        for file_name in dataFileList:
            self.process_file(data_path, file_name)

        self.create_dataframe()

    def process_file(self, data_path, file_name):
        logging.info(f"UPLOAD FILE: {file_name}")
        file_path = os.path.join(data_path, file_name)

        document = fitz.open(file_path)
        page_count = document.page_count
        toc = document.get_toc()
        meta = document.metadata
        logging.info(f"METADATA: {meta}")

        pages = self.extract_pages(document, page_count)
        self.process_toc_or_pages(file_name, toc, pages, page_count)

    def extract_pages(self, document, page_count):
        pages = []
        for page_num in range(page_count):
            page = document.load_page(page_num)
            page_blocks = page.get_text("blocks")
            blocks = "".join(block[4] for block in page_blocks)
            pages.append({"page_num": page_num + 1, "blocks": blocks})
        return pages

    def process_toc_or_pages(self, file_name, toc, pages, page_count):
        if not toc or len(toc) == 0:
            logging.info("Toc Empty")
            for page in pages:
                block = page["blocks"]
                if file_name not in self.doctext_dict:
                    self.doctext_dict[file_name] = [block]
                else:
                    self.doctext_dict[file_name].append([block])
            logging.info("Doc per Created")
        else:
            self.process_with_toc(toc, pages, page_count)

    def process_with_toc(self, toc, pages, page_count):
        for entry in toc:
            level, title, start_page = entry
            end_page = (
                next((e[2] for e in toc if e[2] > start_page), page_count + 1) - 1
            )
            for page in pages:
                block = page["blocks"].replace("\n", " ")
                if start_page <= page["page_num"] <= end_page:
                    if title not in self.doctext_dict:
                        self.doctext_dict[title] = [block]
                    else:
                        self.doctext_dict[title].append([block])
            logging.info("Toc per Created")

    def create_dataframe(self):
        rows = []
        for key, value in self.doctext_dict.items():
            if isinstance(value, list):
                for v in value:
                    rows.append([key, v])
            else:
                rows.append([key, value])

        self.df = pd.DataFrame(rows, columns=["title", "text"])
        self.df["tokens"] = [
            self.num_tokens_from_string(f"{row['title']} {row['text']}")
            for _, row in self.df.iterrows()
        ]

    @staticmethod
    def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def generate_graph_documents(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key
        )
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=["description"],
            relationship_properties=["description"],
        )

        def process_text(text: str) -> List[GraphDocument]:
            doc = Document(page_content=text)
            return llm_transformer.convert_to_graph_documents([doc])

        graph_documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(process_text, f"{row['title']} {row['text']}")
                for _, row in self.df.head(len(self.df)).iterrows()
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                graph_documents.extend(future.result())

        self.graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=True
        )

    def create_vector_store(self):
        vector = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            node_label="__Entity__",
            text_node_properties=["id", "description"],
            embedding_node_property="embedding",
        )
        return vector

    def project_graph(self):
        gds = GraphDataScience(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )

        gds.set_database(self.args.DATABASE)

        G, result = gds.graph.project(
            "entities",  #  Graph name
            "__Entity__",  #  Node projection
            "*",  #  Relationship projection
            nodeProperties=["embedding"],  #  Configuration parameters
        )

        return gds, G

    def perform_knn_and_wcc(self, gds, G, similarity_threshold=0.95):
        gds.knn.write(
            G,
            nodeProperties=["embedding"],
            writeRelationshipType="SIMILAR_KNN",
            writeProperty="score",
            similarityCutoff=similarity_threshold,
        )

        gds.wcc.write(G, writeProperty="wcc", relationshipTypes=["SIMILAR_KNN"])

    def identify_duplicates(self, word_edit_distance=3):
        potential_duplicate_candidates = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 4
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            WITH distinct
              [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                    CASE WHEN index <> index2 AND
                        size(apoc.coll.intersection(acc, results[index2])) > 0
                        THEN apoc.coll.union(acc, results[index2])
                        ELSE acc
                    END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={"distance": word_edit_distance},
        )
        return potential_duplicate_candidates

    @retry(tries=3, delay=2)
    def entity_resolution(self, entities: List[str]) -> Optional[List[str]]:
        system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
        The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

        Here are the rules for identifying duplicates:
        1. Entities with minor typographical differences should be considered duplicates.
        2. Entities with different formats but the same content should be considered duplicates.
        3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
        4. If it refers to different numbers, dates, or products, do not merge results
        """
        user_template = """
        Here is the list of entities to process:
        {entities}

        Please identify duplicates, merge them, and provide the merged list.
        """

        extraction_llm = ChatOpenAI(model_name="gpt-4o").with_structured_output(
            Disambiguate
        )

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_template),
            ]
        )

        extraction_chain = extraction_prompt | extraction_llm

        return [
            el.entities
            for el in extraction_chain.invoke({"entities": entities}).merge_entities
        ]

    def process_duplicate_entities(self, potential_duplicate_candidates):
        merged_entities = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.entity_resolution, el["combinedResult"])
                for el in potential_duplicate_candidates
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                to_merge = future.result()
                if to_merge:
                    merged_entities.extend(to_merge)

        self.graph.query(
            """
        UNWIND $data AS candidates
        CALL {
        WITH candidates
        MATCH (e:__Entity__) WHERE e.id IN candidates
        RETURN collect(e) AS nodes
        }
        CALL apoc.refactor.mergeNodes(nodes, {properties: {
            `.*`: 'discard'
        }})
        YIELD node
        RETURN count(*)
        """,
            params={"data": merged_entities},
        )

    def drop_graph(self, G):
        G.drop()

    def summarize_elements(self, gds, G):
        G, result = gds.graph.project(
            "communities",
            "__Entity__",
            {
                "_ALL_": {
                    "type": "*",
                    "orientation": "UNDIRECTED",
                    "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                }
            },
        )

        wcc = gds.wcc.stats(G)
        print(f"WCC Component count: {wcc['componentCount']}")
        print(f"WCC Component distribution: {wcc['componentDistribution']}")

        gds.leiden.write(
            G,
            writeProperty="communities",
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
        )

        self.graph.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
        )

        self.graph.query(
            """
        MATCH (e:`__Entity__`)
        UNWIND range(0, size(e.communities) - 1 , 1) AS index
        CALL {
        WITH e, index
        WITH e, index
        WHERE index = 0
        MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET c.level = index
        MERGE (e)-[:IN_COMMUNITY]->(c)
        RETURN count(*) AS count_0
        }
        CALL {
        WITH e, index
        WITH e, index
        WHERE index > 0
        MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET current.level = index
        MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
        ON CREATE SET previous.level = index - 1
        MERGE (previous)-[:IN_COMMUNITY]->(current)
        RETURN count(*) AS count_1
        }
        RETURN count(*)
        """
        )

        self.graph.query(
            """
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
        WITH c, count(distinct d) AS rank
        SET c.community_rank = rank;
        """
        )

    def calculate_community_size(self):
        community_size = graph.query(
            """
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
        WITH c, count(distinct e) AS entities
        RETURN split(c.id, '-')[0] AS level, entities
        """
        )
        community_size_df = pd.DataFrame.from_records(community_size)
        return community_size_df

    def calculate_percentiles(self, community_size_df):
        percentiles_data = []
        for level in community_size_df["level"].unique():
            subset = community_size_df[community_size_df["level"] == level]["entities"]
            num_communities = len(subset)
            percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
            percentiles_data.append(
                [
                    level,
                    num_communities,
                    percentiles[0],
                    percentiles[1],
                    percentiles[2],
                    percentiles[3],
                    percentiles[4],
                    max(subset),
                ]
            )

        percentiles_df = pd.DataFrame(
            percentiles_data,
            columns=[
                "Level",
                "Number of communities",
                "25th Percentile",
                "50th Percentile",
                "75th Percentile",
                "90th Percentile",
                "99th Percentile",
                "Max",
            ],
        )
        percentiles_df.to_csv("community_level.csv")


# Run

pipeline = GraphProcessingPipeline()
pipeline.load_data()
pipeline.generate_graph_documents()
vector_store = pipeline.create_vector_store()

gds, G = pipeline.project_graph()
pipeline.perform_knn_and_wcc(gds, G)
potential_duplicates = pipeline.identify_duplicates()
pipeline.process_duplicate_entities(potential_duplicates)
pipeline.drop_graph(G)

community_size_df = pipeline.calculate_community_size()
pipeline.calculate_percentiles(community_size_df)
