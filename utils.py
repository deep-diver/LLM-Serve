from chats import alpaca
from chats import alpaca_gpt4

def get_chat_interface(model_type, batch_enabled):
    if model_type == "alpaca":
        return alpaca.chat_batch if batch_enabled else alpaca.chat_stream
    if model_type == "alpaca-gpt4":
        return alpaca_gpt4.chat_batch if batch_enabled else alpaca_gpt4.chat_stream
    else:
        return None
