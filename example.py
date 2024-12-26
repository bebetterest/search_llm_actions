import os
import sys
import logging
import traceback
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import jsonlines
from tqdm import tqdm

from search_llm_actions import (
    run_shell_script, run_check_vllm,
    Search, Node,
    VLLMCaller, TogetherCaller, try_till_success
)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)


def get_logger(name: str, log_file: str = "main.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, default="search_res.jsonl")
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--expansion_width", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--worker_num", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    logger = get_logger(__name__)
    logger.info("### Start search ###")

    args = get_args()

    # Set llm_caller
    args.llm_caller = TogetherCaller()

    # # deploy model if it's not deployed
    # run_check_vllm()
    
    # Get infos from data
    infos = [
        {
            "tags": {"tag": f"tag{_}"},
            "something": f"something{_}",
        }
        for _ in range(16)
    ]

    td = tqdm(total=len(infos), desc="Total", position=0)
    suc_count = 0
    for batch_start in range(0, len(infos), args.mini_batch_size):
        batch_idx = batch_start // args.mini_batch_size
        with ProcessPoolExecutor(max_workers=args.worker_num) as executor:
            futures = []
            for idx in range(
                batch_start, min(batch_start + args.mini_batch_size, len(infos))
            ):
                args.tags = infos[idx]["tags"]
                search = Search.from_args(args, infos[idx])
                future = executor.submit(search.search, td_position=(idx-batch_start+1))
                futures.append(future)
            for future in as_completed(futures):
                if future.exception() is not None:
                    logger.error(future.exception())
                    logger.error(traceback.format_exc())
                else:
                    reward_ave, dic_res = future.result()
                    with jsonlines.open(args.output_file, mode="a") as writer:
                        writer.write(dic_res)
                    td.update(1)
                    suc_count += any([
                        node["node_info"]["reward_sum"] == 1.0
                        for node in dic_res["nodes"].values()
                        if node["node_info"]["end_flag"]
                    ])
                    td.set_postfix({
                        "batch": batch_idx,
                        "suc_count": suc_count,
                    })
    logger.info("### Finish search ###")


if __name__ == "__main__":
    main()
