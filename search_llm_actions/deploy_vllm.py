import logging
logger = logging.getLogger(__name__)

import os
import sys
import time
import subprocess
from typing import Tuple

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)

SCRIPTS_DIR = os.path.join(current_dir, "scripts")
DEPLOY_VLLM_SH_PATH = os.path.join(SCRIPTS_DIR, "deploy_vllm_multi.sh")
TEST_VLLM_SH_PATH = os.path.join(SCRIPTS_DIR, "test_vllm_multi.sh")

def run_shell_script(script_path) -> Tuple[bool, str]:
    try:
        os.chmod(script_path, 0o755)
        result = subprocess.run(
            ['bash', script_path], 
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error when running {script_path}: {e.stderr}")
        return False, e.stderr


def run_check_vllm(sleep_time: int = 10):
    logger.info("Start checking vllm.")
    flag, res_str = run_shell_script(DEPLOY_VLLM_SH_PATH)
    assert flag, f"Error when running {DEPLOY_VLLM_SH_PATH}: {res_str}"
    logger.info("Vllm is deployed.")

    logger.info("Start testing vllm.")
    while True:
        time.sleep(sleep_time)
        logging.info("Test vllm.")
        flag, res_str = run_shell_script(TEST_VLLM_SH_PATH)
        assert flag, f"Error when running {TEST_VLLM_SH_PATH}: {res_str}"
        if "failed" in res_str:
            logger.info("'failed' is found in the result. Wait for vllm to be ready.")
        else:
            logger.info("Vllm is ready.")
            break


def test_run_check_vllm():
    logger.info("### Start testing run_check_vllm ###")
    run_check_vllm()
    logger.info("### Finish testing run_check_vllm ###")


if __name__ == "__main__":
    test_run_check_vllm()
