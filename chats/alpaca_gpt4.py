import global_vars
from chats import pre
from pingpong import PingPong
from gens.batch_gen import get_output_batch

def wipe_weird_pong_ends(ppmanager):
    last_pong = ppmanager.pingpongs[-1].pong
    last_pong_len = len(last_pong)
    
    tmp_idx = last_pong_len-1
    for char in reversed(last_pong):
        if char in ["!", ".", "?"] \
            and tmp_idx != last_pong_len-1: 
            last_pong = last_pong[:tmp_idx+1]
            break
            
        tmp_idx -= 1

    ppmanager.pingpongs[-1].pong = last_pong
    return ppmanager
    
def strip_pong(ppmanager):
    ppmanager.pingpongs[-1].pong = ppmanager.pingpongs[-1].pong.strip()
    return ppmanager
    
def handle_stream_text(ppmanager, streamer):
    sandbox = ""
    sandbox_enabled = False
    for new_text in streamer:
        new_text = new_text.replace("�", "")
        
        if "###" in new_text:
            sandbox_enabled = True
            sandbox = new_text
        elif "Instruction:" in new_text \
            or "Response:" in new_text \
            or "Comment:" in new_text:
            break
        else:
            if sandbox_enabled:
                sandbox += new_text
                sandbox_enabled = False

            if "### Instruction:" in sandbox or \
                "### Response:" in sandbox or \
                "### Input:" in sandbox:
                break
            else:
                ppmanager.append_pong(new_text)
                yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()
    
def chat_stream(user_message, state):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    # add_pong("") means no response yet (to avoid None). Later, tokens will be appended
    prompt = ppm.add_ping(user_message)
    ppm.add_pong("")
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build_pipeline(prompt, global_vars.gen_config)
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in handle_stream_text(ppm, streamer):
        yield "", uis, state

    ppm = strip_pong(ppm)
    ppm = wipe_weird_pong_ends(ppm)
    state["ppmanager"] = ppm
    yield "", ppm.build_uis(), state
    
    # summarization
    ppm.add_pingpong(
        PingPong(None, "![](https://s2.gifyu.com/images/icons8-loading-circle.gif)")
    )
    yield "", ppm.build_uis(), state
    
    # ppm.pop_pingpong()
    # ppm.
