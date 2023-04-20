import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def build_prompt(ppmanager, user_message, win_size=2):
    dummy_ppm = copy.deepcopy(ppmanager)
    dummy_ppm.pop_pingpong()
    lws = CtxLastWindowStrategy(win_size)
    
    lws_result = lws(dummy_ppm)
    prompts = dummy_ppm.add_ping(user_message)
    print(f"lws_result: {lws_result}")
    print(f"prompts: {prompts}")
    prompts = prompts.replace(
        system_prompt,
        system_prompt+lws_result
    )
    return prompts

def text_stream(ppmanager, streamer):
    for new_text in streamer:
        ppmanager.append_pong(new_text)
        yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()

def chat_stream(user_message, state):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompt(ppm, user_message)
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(
        prompt, global_vars.gen_config_raw, StoppingCriteriaList([StopOnTokens()])
    )
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        ppm = ppmanager
        yield "", uis, prompt, state

    state["ppmanager"] = ppm
    yield "", uis, prompt, state
