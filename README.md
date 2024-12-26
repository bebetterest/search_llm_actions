# search_llm_actions

search_llm_actions provides a simple template to collect action trajectories from local (vllm servers on multi-gpus are supported) or remote (togetherai api is supported) llms by search.

## Installation

```bash
pip install search_llm_actions
```

## Customization

- You could find an minimal example in `search_llm_actions/main.py`.
- You need to override  `init_root_node`, `take_parallel_actions`, `simulate` & `detect_end` functions in `search_llm_actions/search.py` to adapt to your own task.
- You need to override `deploy_vllm_multi.sh` & `test_vllm_multi.sh` in `search_llm_actions/scripts` to adapt to your own llm server.
- You need to implement a new subclass of `Caller` in `search_llm_actions/llm_caller.py` to adapt to your own llm server.

enjoy:)
🤯
