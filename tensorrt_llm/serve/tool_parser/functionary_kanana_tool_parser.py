import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import partial_json_parser
from partial_json_parser.core.options import Allow

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc
from .utils import find_common_prefix, is_complete_json, partial_json_loads


class BaseTemplate(ABC):
    @abstractmethod
    def response_to_messages(self, generated_text):
        raise NotImplementedError


class FunctionaryV3Llama31Template(BaseTemplate):
    def __init__(self):
        self.system_tokens = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_tokens = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_tokens = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.tool_tokens = "<|start_header_id|>ipython<|end_header_id|>\n\n"

    def parse_function_call_from_text(self, function_call_text: str) -> Optional[Dict]:
        index = function_call_text.find(">")
        if index >= 0:
            func_name = function_call_text[:index].strip()
            arguments = function_call_text[index + 1 :].strip()
            return {"name": func_name, "arguments": arguments}
        return None

    def response_to_messages(self, generated_text):
        # first remove stop tokens if there exists
        for stop in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"]:
            if generated_text.endswith(stop):
                generated_text = generated_text[: -len(stop)]

        tool_calls = []
        text_response = ""

        func_prefix = "<function="
        end_func = "</function>"
        python_tag = "<|python_tag|>"

        while len(generated_text) > 0:
            if generated_text.startswith(python_tag):  # check if use code interpreter
                code = generated_text[len(python_tag) :]
                function_call = {
                    "name": "python",
                    "arguments": code,
                }

                tool_calls.append(function_call)
                generated_text = ""
            elif generated_text.startswith(func_prefix):  # Check if function_call
                end_index = generated_text.find(end_func)
                if end_index >= 0:
                    function_call_text = generated_text[len(func_prefix) : end_index]
                    function_call = self.parse_function_call_from_text(function_call_text)

                    tool_calls.append(function_call)
                    generated_text = generated_text[end_index + len(end_func) :]
                else:
                    # TODO cannot find close function call
                    text_response += generated_text
                    break
            else:  # If text-response
                text_response += generated_text[0]
                generated_text = generated_text[1:]

        if not text_response:
            text_response = ""
        elif len(text_response.strip()) == 0:
            text_response = ""

        if tool_calls:
            return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}
        else:
            return {"role": "assistant", "content": text_response}


class FunctionaryV3Llama31ToolParser(BaseToolParser):
    def __init__(self):
        super().__init__()
        self.template = FunctionaryV3Llama31Template()

        self._func_prefix = "<function="
        self._func_suffix = "</function>"
        self._python_tag = "<|python_tag|>"
        self.current_tool_name_sent: bool = False
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: List[
            str
        ] = []  # map what has been streamed for each tool so far to a list
        self.eos_token = "<|eot_id|>"
        # added buffer for each tool call parser
        self._buffer = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Functionary V3 Llama 3.1 format tool call."""
        return self._python_tag in text or self._func_prefix in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        result = self.template.response_to_messages(text)
        text = result.get("content", "")
        for stop in [self.eos_token, "<|eom_id|>", "<|end_of_text|>"]:
            if text.endswith(stop):
                text = text[: -len(stop)]
        calls = self.parse_base_json(result.get("tool_calls", []), tools)
        return StreamingParseResult(normal_text=text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer
        delta_text = new_text
        # if current_text does not start with function tag (or python tag),
        if not (
            current_text.startswith(self._python_tag)
            or current_text.startswith(self._func_prefix)
            or self._func_prefix.startswith(current_text)
        ):
            # for cases like "The answer is <function="
            # let current_text="<function="
            if self._func_prefix in current_text:
                idx = current_text.find(self._func_prefix)
                self._buffer = current_text[idx:]
                return StreamingParseResult(normal_text=current_text[:idx])
            # for cases like "The answer is <function"
            # add delta_text to buffer to figure out whether to print or not later
            elif delta_text.endswith("<") or (
                current_text.endswith("<function") and delta_text.endswith("function")
            ):
                return StreamingParseResult()
            # for cases that does not include "<function" at all,
            # stream right away as delta.content
            else:
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)

        # CHECK: this part not working (should use current_token_ids)
        # if current_text ends with stop token,
        # remove it from the text
        # CHECK: sometimes text is generated beyond <|eom_id|>
        for stop in [self.eos_token, "<|eom_id|>", "<|end_of_text|>"]:
            if current_text.rstrip().endswith(stop):
                current_text = current_text.rstrip()[: -len(stop)]

        # if current_tool_name is not sent yet,
        # don't allow partial sending of strings
        # (supposedly, openai also only sends the entire tool name at once)
        # not really relevant to kanana
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            tool_call_arr = []
            is_complete = []
            obj = {}
            try:
                # set start_idx
                start_idx = 0
                while start_idx < len(current_text):
                    name = None
                    # set function_name and move start_idx to the end of function_name
                    if current_text[start_idx:].startswith(self._python_tag):
                        name = "python"
                        start_idx += len(self._python_tag)
                    elif current_text[start_idx:].startswith(self._func_prefix):
                        idx = current_text[start_idx:].find(">")
                        # found ">" which means function name is ready
                        if idx != -1:
                            function_name = current_text[
                                start_idx + len(self._func_prefix) : start_idx + idx
                            ]
                            name = function_name
                            start_idx += len(self._func_prefix) + len(function_name) + len(">")
                    # for cases like: </function>abcd<function=
                    # ignore tokens in between </function> and <function=
                    # and move start_idx to the start of the new function
                    else:
                        idx = current_text[start_idx:].find(self._func_prefix)
                        if idx != -1:
                            start_idx += idx
                            continue

                    # partially load the function arguments
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    if "arguments" not in obj:
                        obj["arguments"] = json.loads(json.dumps(obj, ensure_ascii=False))
                    if name is not None:
                        obj["name"] = name

                    is_complete.append(
                        is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )

                    start_idx += end_idx

                    # TODO: is this necessary for kanana? can we remove this?
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert "arguments" not in obj, (
                            "model generated both parameters and arguments"
                        )
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

                    # if you can find </function>, which declares the end of a function,
                    # move start_idx to the end so it can skip </function>
                    function_end_idx = current_text[start_idx:].find(self._func_suffix)
                    if function_end_idx != -1:
                        start_idx += len(self._func_suffix)
                    # if </function> hasn't been generated fully yet,
                    # but the function is complete, we break out from while loop
                    # and stream it right away
                    elif is_complete[-1]:
                        break
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return StreamingParseResult()

            # current_tool_call is the one that is being streamed
            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case0: if there is no tool call in the array, return None
            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            # case1: we are starting a new tool in the array
            #   -> tool_call_arr has > 0 length AND has more elements than cursor
            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                # print("starting a new tool in the array, print remaining")
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(
                            self.streamed_args_for_tool[self.current_tool_id]
                        )  # streamed args for current tool call
                        argument_diff = cur_args_json[
                            sent:
                        ]  # args not yet streamed for current tool call

                        logger.debug("got arguments diff: %s", argument_diff)
                        delta = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=argument_diff,
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff  # update streamed args for current tool call
                        )
                    else:
                        delta = StreamingParseResult()
                else:
                    delta = StreamingParseResult()
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1  # update current tool call
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # case2: if the current tool name hasn't been sent, send if available
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = StreamingParseResult()

            # case3: now we know we're on the same tool call
            # and can stream arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = StreamingParseResult()
                sent = len(
                    self.streamed_args_for_tool[self.current_tool_id]
                )  # streamed args for current tool call
                cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                argument_diff = None
                # if current tool call is complete
                if is_complete[self.current_tool_id]:
                    self._buffer = current_text[start_idx:]
                    argument_diff = cur_args_json[sent:]
                # if current tool call is not complete and prev_arguments exists
                elif prev_arguments:
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    if cur_args_json != prev_args_json:
                        # common prefix betwn prev and current args
                        prefix = find_common_prefix(prev_args_json, cur_args_json)
                        argument_diff = prefix[sent:]
                if argument_diff is not None and len(argument_diff) > 0:
                    delta = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters=argument_diff,
                            )
                        ]
                    )

                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            # update previous tool call array
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<function=" + name + ">",
            end="</function>",
            trigger="<function=",
        )
