import os
import sys
import time
import random
import inspect
import logging
import requests
import traceback
import jsonlines
from copy import deepcopy
from typing import List, Dict, Union

from together import Together
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)

from dotenv import load_dotenv
load_dotenv()


def get_logger(name: str, log_file: str = "caller.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class Caller():
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def call(
            self,
            messages: List[Dict[str, str]],
            n: int = 1,
    ) -> List[str]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        keys = inspect.getfullargspec(self.call).args[:len(args)]
        keyyed_args = {
            k: v
            for k, v in zip(keys, args)
        }
        keyyed_args.update(kwargs)
        result = self.call(*args, **kwargs)
        self.logger.info(f"\n-v-args-v-\n{keyyed_args}\n-v-result-v-\n{result}")
        return result


class VLLMCaller(Caller):
    def __init__(
            self,
            api_url: str,
            chat_template_name: str,
            model_name: str = "local_model",
            sampling_params: dict = {
                # "use_beam_search": True,
                # "top_p": 0.95,
                # "temperature": 1.0,
                "max_tokens": 2048,
                "stream": False,
            },
    ):
        super().__init__()
        self.api_url_list = [api_url]
        self.chat_template_name = chat_template_name
        self.model_name = model_name
        self.sampling_params = {
            # "use_beam_search": True,
            "max_tokens": 2048,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
        self.sampling_params.update(sampling_params)

        if os.path.exists("server_ports.txt"):
            print("overriding api_url_list with server_ports.txt")
            with open("server_ports.txt", "r") as f:
                self.api_url_list = [
                    f"http://localhost:{port.strip()}/v1/completions"
                    for port in f.read().strip().split(" ")
                ]
        print(f"api_url_list({len(self.api_url_list)}): {self.api_url_list}")

    def chat_format_messages_by_hf(
            self,
            chat_template_name: str,
            messages: List[Dict[str, str]],
    ) -> str:
        tokenizer = AutoTokenizer.from_pretrained(chat_template_name)
        if messages[-1]['role'] == 'assistant':
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt + messages[-1]['content']
        else:
            assert messages[-1]['role'] == 'user'
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt

    def parse_response(
            self,
            response,
    ) -> List[str]:
        text_res = []
        for choice in response.json()['choices']:
            text_res.append(choice['text'])
        return text_res

    def url_select(self):
        selected_url = None
        requests_num = None
        all_same_flag = True
        for url in self.api_url_list:
            tmp = requests.get(url.replace("v1/completions", "metrics")).text
            running_num = float(
                tmp.split('vllm:num_requests_running{model_name="local_model"}')[1].split("\n")[0].strip()
            )
            swapped_num = float(
                tmp.split('vllm:num_requests_swapped{model_name="local_model"}')[1].split("\n")[0].strip()
            )
            waiting_num = float(
                tmp.split('vllm:num_requests_waiting{model_name="local_model"}')[1].split("\n")[0].strip()
            )
            total_num = running_num + swapped_num + waiting_num
            if requests_num == None or total_num < requests_num:
                requests_num = total_num
                selected_url = url
            if requests_num != None and total_num != requests_num:
                all_same_flag = False
        if all_same_flag:
            return random.choice(self.api_url_list)
        else:
            return selected_url

    def call(
            self,
            messages: List[Dict[str, str]],
            n: int = 1,
    ) -> List[str]:
        headers = {"User-Agent": "ChatTest Client"}
        prompt = self.chat_format_messages_by_hf(
            chat_template_name=self.chat_template_name,
            messages=messages,
        )
        sampling_params = deepcopy(self.sampling_params)
        sampling_params.update({
            "n": n,
        })
        pload = {
            "model": self.model_name,
            "prompt": prompt,
            **sampling_params,
        }
        response = requests.post(
            self.url_select(),
            headers=headers,
            json=pload
        )
        text_res = self.parse_response(response)
        return text_res


class TogetherCaller(Caller):
    def __init__(
            self,
            api_base: Union[str, None] = None,
            api_key_list: List[str] =\
                [
                    _.strip()
                    for _ in os.environ["TOGETHER_API_KEY"].strip().split(" ")
                ],
            engine: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            sampling_params: Dict = {
                "max_tokens": 2048,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
            },
    ):
        super().__init__()
        self.api_base = api_base
        self.api_key_list = api_key_list
        self.engine = engine
        self.sampling_params = {
            "max_tokens": 2048,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
        self.sampling_params.update(sampling_params)

        client_params = {}
        if self.api_base is not None:
            client_params["base_url"] = self.api_base
        self.clients = [
            Together(api_key=api_key, **client_params)
            for api_key in self.api_key_list
        ]

    def call(
            self,
            messages: List[Dict[str, str]],
            n: int = 1,
    ) -> List[str]:
        client = random.choice(self.clients)
        response = client.chat.completions.create(
            model=self.engine,
            messages=messages,
            **self.sampling_params,
            n=n,
        )
        if response.choices[0].finish_reason == "length":
            raise Exception("Response length exceeds max_tokens")
        res = [
            choice.message.content.strip()
            for choice in response.choices
        ]
        return res


def try_till_success(
        func,
        try_tags=None,
        max_tries=4096, max_try_time=180, try_interval=2,
        *args, **kwargs,
):
    start_time = time.time()
    for i in range(max_tries):
        try:
            return func(*args, **kwargs), try_tags
        except Exception as e:
            print(f"###\ERROR:\n{e}\n{traceback.format_exc()}\n###")
            if (time.time() - start_time) > max_try_time:
                raise f"###\nExceed max try time {max_try_time}\n###"
            time.sleep(try_interval)
    raise f"###\nExceed max tries {max_tries}\n###"


def test_vllmcaller():
    api_url = "http://localhost:8000/v1/completions"
    vllm_caller = VLLMCaller(
        api_url=api_url,
        chat_template_name="Qwen/QwQ-32B-Preview",
    )
    messages = [
        {"role": "assistant", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": "Introduce yourself."},
    ]
    response, _ = try_till_success(
        vllm_caller,
        messages=messages,
        n=1,
    )
    print(f"response:\n{response}")


def test_togethercaller():
    together_caller = TogetherCaller()
    messages = [
        {"role": "assistant", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": "Introduce yourself."},
    ]
    response, _ = try_till_success(
        together_caller,
        messages=messages,
        n=1,
    )
    print(f"response:\n{response}")


if __name__ == "__main__":
    # print("###\nStart testing VLLMCaller\n###")
    # test_vllmcaller()
    print("###\nStart testing TogetherCaller\n###")
    test_togethercaller()
