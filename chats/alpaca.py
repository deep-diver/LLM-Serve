import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

def build_prompt(ppmanager, user_message, win_size=2):
    dummy_ppm = copy.deepcopy(ppmanager)
    dummy_ppm.pop_pingpong()
    lws = CtxLastWindowStrategy(win_size)
    
    lws_result = lws(dummy_ppm)
    lws_result = lws_result.replace("### Instruction:\n", "<Human> ")
    lws_result = lws_result.replace("### Response:\n", "<LLaMA> ")
    if lws_result != "":
        lws_result = f''' Below is recent conversation between you(LLaMA) and me from top to bottom. Take them into account, but do not repeat as is.
"{lws_result}"
'''
    
    dummy_ppm.ctx += lws_result
    prompts = dummy_ppm.add_ping(user_message)    
    return prompts

def text_stream(ppmanager, streamer):
    sandbox = ""
    sandbox_enabled = False
    for new_text in streamer:
        ppmanager.append_pong(new_text)
        yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()

def summarize(ppmanager):
    ctx = ppmanager.ctx
    pong = ppmanager.pingpongs[-1].pong
    if ctx is None or ctx == "":
        ping = f'given the context of "{ctx}", extract context from the <LLaMA>\'s response "{pong}". be clear about who is LLaMA and its response.'
    else:
        ping = f'summarize "{pong}"'
    prompt = ppmanager.add_ping(ping)
    
    summarize_output = get_output_batch(
        global_vars.model, global_vars.tokenizer, [prompt], global_vars.gen_config_summarization
    )[0].split("### Response:")[-1].strip()
    ppmanager.ctx = summarize_output
    ppmanager.pop_pingpong()
    return ppmanager
    
def chat_stream(user_message, state):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompt(ppm, user_message)
    print(prompt)
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(prompt, global_vars.gen_config)
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        yield "", uis, state

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), state
    
    # summarization
    ppm.add_pingpong(
        PingPong(None, "![](https://s2.gifyu.com/images/icons8-loading-circle.gif)")
    )
    yield "", ppm.build_uis(), state
    ppm.pop_pingpong()
    
    ppm = summarize(ppm)
    state["ppmanager"] = ppm
    yield "", ppm.build_uis(), state