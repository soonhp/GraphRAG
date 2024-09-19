import os
from parser import parse_args
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


args = parse_args()


class Neo4jEnvironment:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.args = parse_args()

    def set_environment(self):
        os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
        os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
        os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

    def create_database(self):
        graph = Neo4jGraph()
        graph.query(
            "create database $database", params={"database": self.args.DATABASE}
        )
        os.environ["NEO4J_DATABASE"] = self.args.DATABASE


neo4j_env = Neo4jEnvironment()
neo4j_env.set_environment()
neo4j_env.create_database()
