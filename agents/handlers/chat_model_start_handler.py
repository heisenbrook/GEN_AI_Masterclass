from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage

class ChatModelStartHandler(BaseCallbackHandler):
    '''
    Docstring for ChatModelStartHandler
    Generates a callback - useful for debugging, no need for verbose=True anymore.
    '''
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any):
        print(messages)