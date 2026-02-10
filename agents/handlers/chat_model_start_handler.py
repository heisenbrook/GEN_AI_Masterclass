from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from pyboxen import boxen

def boxen_print(*args, **kwargs):
    '''
    Docstring for boxen_print
    Helper function for output text formatting
    Accept a message content (str), a title and a color
    '''
    print(boxen(*args, **kwargs))

class ChatModelStartHandler(BaseCallbackHandler):
    '''
    Docstring for ChatModelStartHandler
    Generates a callback - useful for debugging, no need for verbose=True anymore.
    '''
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any):
        print("\n\n\n\n========= Sending Messages =========\n\n")

        for message in messages[0]:
            # Adding a different format for every tipe of message we can get
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")
            
            elif message.type == "ai" and "function_call" in message.additional_kwargs: 
                call = message.additional_kwargs["function_call"]
                boxen_print(f"Running tool {call['name']} with args {call['arguments']}",
                            title=message.type,
                            color="cyan"
                            )
            
            elif message.type == "ai":
                # Only format plain AI messages and not every piece of text generated
                boxen_print(message.content, title=message.type, color="blue")
            
            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")

            else:
                boxen_print(message.content, title=message.type, color="blue")
