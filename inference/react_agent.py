import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *
from hf_spaces_client import create_client
from tool_logger import ToolLogger

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 tool_logger: Optional[ToolLogger] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.tool_logger = tool_logger

        # Initialize API client
        # Check if we should use OpenRouter or HF Spaces
        use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
        self.api_client = create_client(use_openrouter=use_openrouter)
        print(f"✅ API client initialized: {'OpenRouter' if use_openrouter else 'HF Spaces'}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):
        """
        Call the API server (HF Spaces or OpenRouter) with retry logic.

        Args:
            msgs: List of messages to send
            planning_port: Port number (kept for compatibility, unused for API)
            max_tries: Maximum retry attempts

        Returns:
            Response content string
        """

        try:
            print(f"--- Calling API service ---")

            # Use the initialized API client with its built-in retry logic
            response = self.api_client.chat_completion(
                messages=msgs,
                temperature=self.llm_generate_cfg.get('temperature', 0.85),
                top_p=self.llm_generate_cfg.get('top_p', 0.95),
                max_tokens=10000,
                presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                stop=["\n<tool_response>", "<tool_response>"],
                max_retries=max_tries
            )

            # Extract content using client method
            content = self.api_client.extract_content(response)

            if content and content.strip():
                print("--- Service call successful, received a valid response ---")
                return content.strip()
            else:
                print("Warning: Received an empty response.")
                return "API server error: empty response"

        except Exception as e:
            print(f"Error: API call failed after all retries: {e}")
            return f"API server error: {str(e)}"

    def count_tokens(self, messages):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        
        return token_count

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                # Write tool log summary
                if self.tool_logger:
                    self.tool_logger.write_summary(question, answer, prediction, termination, round)
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            # Log Python tool call
                            python_start = time.time()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                            python_duration = time.time() - python_start
                            if self.tool_logger:
                                self.tool_logger.log_tool_call(
                                    round_num=round,
                                    tool_name="PythonInterpreter",
                                    tool_input={"code": code_raw},
                                    tool_output=result,
                                    duration=python_duration,
                                    success=True,
                                    error=None
                                )
                        except Exception as e:
                            result = "[Python Interpreter Error]: Formatting error."
                            if self.tool_logger:
                                self.tool_logger.log_tool_call(
                                    round_num=round,
                                    tool_name="PythonInterpreter",
                                    tool_input={"code": tool_call},
                                    tool_output=result,
                                    duration=0,
                                    success=False,
                                    error=str(e)
                                )

                    else:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args, round_num=round)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                # Write tool log summary
                if self.tool_logger:
                    self.tool_logger.write_summary(question, answer, prediction, termination, round)
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        # Write tool log summary
        if self.tool_logger:
            self.tool_logger.write_summary(question, answer, prediction, termination, round)
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, round_num: int = 0, **kwargs):
        start_time = time.time()
        error_msg = None
        success = True

        if tool_name in TOOL_MAP:
            try:
                tool_args["params"] = tool_args
                if "python" in tool_name.lower():
                    result = TOOL_MAP['PythonInterpreter'].call(tool_args)
                elif tool_name == "parse_file":
                    params = {"files": tool_args["files"]}

                    raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                    result = raw_result

                    if not isinstance(raw_result, str):
                        result = str(raw_result)
                else:
                    raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                    result = raw_result
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
                success = False
                error_msg = str(e)
        else:
            result = f"Error: Tool {tool_name} not found"
            success = False
            error_msg = "Tool not found"

        duration = time.time() - start_time

        # Log the tool call if logger is available
        if self.tool_logger:
            self.tool_logger.log_tool_call(
                round_num=round_num,
                tool_name=tool_name,
                tool_input=tool_args,
                tool_output=result,
                duration=duration,
                success=success,
                error=error_msg
            )

        return result
