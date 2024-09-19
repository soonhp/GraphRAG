# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/parser.py>.

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Community summary by level")
    parser.add_argument(
        "--DATABASE", type=str, default="", help="NEO4J_DATABASE_setting"
    )
    parser.add_argument(
        "--level1", type=int, default=0, help="community level1 setting"
    )
    parser.add_argument(
        "--level2", type=int, default=1, help="community level2 setting"
    )
    parser.add_argument(
        "--level3", type=int, default=4, help="community level3 setting"
    )
    parser.add_argument(
        "--index_name", type=str, default="", help="NEO4J_VECTOR_INDEX_NAME"
    )
    parser.add_argument("--question", type=str, default="", help="User_Query")
    parser.add_argument(
        "--DATA_PATH",
        type=str,
        default="/home/infocz/soonhyeok/GoRAG/gorag_CommunityDetection/KBlife/data",
        help="DATA_PATH(Directory)",
    )
    return parser.parse_args()
