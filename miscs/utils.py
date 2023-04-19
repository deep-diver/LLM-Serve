import re

from miscs.strings import SPECIAL_STRS
from miscs.constants import html_tag_pattern, multi_line_pattern, multi_space_pattern
from miscs.constants import repl_empty_str, repl_br_tag, repl_span_tag_multispace, repl_linebreak

# applicable to instruction to be displayed as well
def common_post_process(original_str):
    original_str = re.sub(
        multi_line_pattern, repl_br_tag, original_str
    )
    original_str = re.sub(
        multi_space_pattern, repl_span_tag_multispace, original_str
    )
    
    return original_str

def post_process_stream(bot_response):
    # sometimes model spits out text containing 
    # "### Response:" and "### Instruction: -> in this case, we want to stop generating
    if "### Response:" in bot_response or "### Input:" in bot_response:
        bot_response = bot_response.replace("### Response:", '').replace("### Input:", '').strip()
        return bot_response, True
    
    return common_post_process(bot_response), False

def post_process_batch(bot_response):
    bot_response = bot_response.split("### Response:")[-1].strip()
    return common_post_process(bot_response)

def post_processes_batch(bot_responses):
    return [post_process_batch(r) for r in bot_responses]