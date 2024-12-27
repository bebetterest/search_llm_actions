import logging
logger = logging.getLogger(__name__)

import os
import sys
import math
import random
import argparse
from copy import deepcopy
from typing import Union, List, Dict, Any, Tuple, Optional

import jsonlines
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)

from llm_caller import Caller, VLLMCaller, TogetherCaller, try_till_success


INIT_NODE_INFO:dict = {
    "parents_idss": [],
    "children_idss": [],
    "visit_count": 0,
    "reward_sum": 0,
    "end_flag": False,
}

INIT_NODE_STAT:dict = {
    "content": None,
}


class Node():
    def __init__(
            self, 
            ids: str,
            node_info: Optional[dict] = None,
            node_stat: Optional[dict] = None,
            init_node_info: dict = INIT_NODE_INFO,
            init_node_stat: dict = INIT_NODE_STAT,
    ):
        self.ids = ids
        self.node_info = deepcopy(init_node_info)
        if node_info is not None:
            self.node_info.update(node_info)
        self.node_stat = deepcopy(init_node_stat)
        if node_stat is not None:
            self.node_stat.update(node_stat)

    def to_json(self):
        return {
            "ids": self.ids,
            "node_info": self.node_info,
            "node_stat": self.node_stat,
        }


class Search():
    def __init__(
            self,
            args: argparse.Namespace,
            tags: Dict[str, str], info: dict,
            llm_caller: Caller, max_iter: int, expansion_width: int, max_depth: int
    ):
        self.args = args
        self.tags = tags
        self.info = info
        self.llm_caller = llm_caller
        self.max_iter = max_iter
        self.expansion_width = expansion_width
        self.max_depth = max_depth

        root_node = self.init_root_node(args, info)
        self.root_ids = root_node.ids
        self.nodes = {self.root_ids: root_node}

    @classmethod
    def from_args(cls, args: argparse.Namespace, info: dict) -> "Search":
        return cls(
            args=args,
            info=info, tags=info.get("tags", {}),
            llm_caller=args.llm_caller,
            max_iter=args.max_iter,
            expansion_width=args.expansion_width,
            max_depth=args.max_depth,
        )
    
    def init_root_node(self, args: argparse.Namespace, info: dict) -> Node:
        # You have to override this function.
        return Node(
            ids="0",
            node_stat={
                "content": "This is the root node.",
            }
        )


    def take_parallel_actions(self, node_ids: str, width: int) -> List[Dict[str, Any]]:
        # You have to override this function.
        messages = [{
            "role": "user",
            "content": f"Hi, feel free to output one short sentence something about {node_ids}. The output has to start with '{node_ids}'",
        }]
        responses = try_till_success(
            self.llm_caller, messages=messages, n=width
        )
        return [
            {
                "content": response,
            }
            for response in responses
        ]
    
    def simulate(self, node_ids: str) -> float:
        # You have to override this function.
        if random.random() > 0.5:
            return 1.0
        else:
            return 0.0

    def detect_end(self, node_ids: str, max_depth: int) -> bool:
        # You have to override this function.
        node = self.nodes[node_ids]
        depth = len(node_ids.split("."))
        if depth >= max_depth:
            return True
        else:
            return False

    def select_node(self, node_ids: str, excluded_idss: list = []) -> Tuple[str, bool]:
        # select a expandable leaf node by UCB1
        if node_ids in excluded_idss:
            return node_ids, False
        if self.nodes[node_ids].node_info["end_flag"] == True:
            return node_ids, False
        if len(self.nodes[node_ids].node_info["children_idss"]) == 0:
            return node_ids, True

        node = self.nodes[node_ids]
        ids_score_dict = {}
        for child_ids in self.nodes[node_ids].node_info["children_idss"]:
            child_node = self.nodes[child_ids]
            if child_ids in excluded_idss or child_node.node_info["end_flag"] == True:
                continue
            if child_node.node_info["visit_count"] == 0:
                score = float("inf")
            else:
                expliotation = child_node.node_info["reward_sum"] / child_node.node_info["visit_count"]
                exploration = math.sqrt(2 * math.log(node.node_info["visit_count"]) / child_node.node_info["visit_count"])
                score = expliotation + exploration
            ids_score_dict[child_ids] = score
        if len(ids_score_dict) == 0:
            return node_ids, False
        sorted_ids_score_dict = sorted(ids_score_dict.items(), key=lambda x: -x[1])
        for ids, _ in sorted_ids_score_dict:
            next_node_ids, flag = self.select_node(ids, excluded_idss)
            if flag:
                return next_node_ids, True
        return node_ids, False

    def expansion(self, node_ids, expansion_width):
        node = self.nodes[node_ids]
        assert node.node_info["end_flag"] == False, "The node is an end node, so it can't be expanded."
        parallel_children_stats = self.take_parallel_actions(node_ids, expansion_width)
        for i, child_stat in enumerate(parallel_children_stats):
            child_ids = f"{node_ids}.{i}"
            child_node = Node(child_ids, node_stat=child_stat)
            self.nodes[child_ids] = child_node
            node.node_info["children_idss"].append(child_ids)
            child_node.node_info["parents_idss"].append(node_ids)
            child_node.node_info["end_flag"] = self.detect_end(child_ids, self.max_depth)
        
    def backpropagation(self, node_ids: str, reward: float):
        node = self.nodes[node_ids]
        node.node_info["visit_count"] += 1
        node.node_info["reward_sum"] += reward
        for parent_ids in node.node_info["parents_idss"]:
            self.backpropagation(parent_ids, reward)
    
    def search_func(
            self,
            node_ids: str, max_iter: int, expansion_width: int,
            td_flag: bool = True, td_position: Optional[int] = None
    ):
        if td_flag:
            if td_position is not None:
                td = tqdm(range(max_iter), desc=f"Search", position=td_position)
            else:
                td = tqdm(range(max_iter), desc="Search")
        else:
            td = range(max_iter)
        for _ in td:
            selected_node_ids, flag = self.select_node(node_ids)
            if not flag:
                logger.info("No expandable leaf node is found.")
                break
            while True:
                self.expansion(selected_node_ids, expansion_width)
                for child_ids in self.nodes[selected_node_ids].node_info["children_idss"]:
                    if self.nodes[child_ids].node_info["end_flag"]:
                        reward = self.simulate(child_ids)
                        self.backpropagation(child_ids, reward)
                selected_node_ids = random.choice(self.nodes[selected_node_ids].node_info["children_idss"])
                if self.nodes[selected_node_ids].node_info["end_flag"]:
                    break
            if td_flag:
                td.set_postfix({
                    "reward_sum": self.nodes[self.root_ids].node_info["reward_sum"],
                    "visit_count": self.nodes[self.root_ids].node_info["visit_count"]
                })
        
    def search(self, td_position: Optional[int] = None) -> Tuple[float, dict]:
        self.search_func(self.root_ids, self.max_iter, self.expansion_width, td_position=td_position)
        return self.nodes[self.root_ids].node_info["reward_sum"] / self.nodes[self.root_ids].node_info["visit_count"], self.to_dict()

    def to_dict(self) -> dict:
        return {
            "args": {
                k: v
                for k, v in vars(self.args).items()
                if not isinstance(v, Caller)
            },
            "tags": self.tags,
            "info": self.info,
            "root_ids": self.root_ids,
            "nodes": {
                ids: node.to_json()
                for ids, node in self.nodes.items()
            },
        }


def test_search():
    args = argparse.Namespace(
        llm_caller=TogetherCaller(),
        max_iter=10,
        expansion_width=2,
        max_depth=5,
    )
    info = {
        "tags": {},
    }
    search = Search.from_args(args, info)
    reward_ave = search.search()
    logger.info(f"reward_ave: {reward_ave}")
    res = search.to_dict()
    logger.info(f"saving search result to search_test_res.jsonl")
    with jsonlines.open("search_test_res.jsonl", "w") as f:
        f.write(res)
    logger.info(f"search result is saved.")


if __name__ == "__main__":
    logger.info("### Start testing search ###")
    test_search()
