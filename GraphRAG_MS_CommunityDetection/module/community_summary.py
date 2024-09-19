import os

from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from parser import parse_args
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

args = parse_args()
# %% Environment Setting
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_DATABASE"] = args.DATABASE
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph()


# %% LLM setting
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

community_info = graph.query(
    """
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level IN [$level1, $level2, $level3]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""",
    params={"level1": args.level1, "level2": args.level2, "level3": args.level3},
)


community_template = """Based on the provided nodes and relationships that belong to the same graph community,
generate a natural language summary of the provided information:
{community_info}

Summary:"""  # noqa: E501

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input triples, generate the information summary. No pre-amble.",
        ),
        ("human", community_template),
    ]
)

community_chain = community_prompt | llm | StrOutputParser()


def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data["nodes"]:
        node_id = node["id"]
        node_type = node["type"]
        if "description" in node and node["description"]:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data["rels"]:
        start = rel["start"]
        end = rel["end"]
        rel_type = rel["type"]
        if "description" in rel and rel["description"]:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str


def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({"community_info": stringify_info})
    return {"community": community["communityId"], "summary": summary}


summaries = []
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(process_community, community): community
        for community in community_info
    }

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing communities"
    ):
        summaries.append(future.result())


# Store summaries
graph.query(
    """
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.summary = row.summary
""",
    params={"data": summaries},
)
