from chats import alpaca
from chats import alpaca_gpt4

def get_chat_interface(model_type):
    if model_type == "alpaca":
        return alpaca.chat_stream
    if model_type == "alpaca-gpt4":
        return alpaca_gpt4.chat_stream
    else:
        return None
