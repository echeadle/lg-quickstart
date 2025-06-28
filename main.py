import io
from PIL import Image
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

# The function load_dotenv() loads enviroment varibles
load_dotenv()


class State(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# The init_chat_model, seems very basic, it does not take a key
# as an argument.
llm = init_chat_model(model="gpt-4o-mini")


def chatbot(state: State):
    """
    A node that invokes the chatbot to respond to the user's message.
    """
    return {"messages": [llm.invoke(state["messages"])]}

def draw_graph(graph):
    """
    Generates and displays a diagram of the graph.
    """
    try:
        img_data = graph.get_graph().draw_mermaid_png()
        image = Image.open(io.BytesIO(img_data))
        image.show()
    except Exception as e:
        # This requires some extra dependencies and is optional.
        # Added error printing for better feedback.
        print(f"Error drawing graph: {e}")

def stream_graph_updates(user_input: str):
    """
    Streams the graph's response to the user's input.
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)



# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def main():
    """
    Main loop to run the chatbot from the command line.
    """
    print("Chatbot is ready. Type 'quit', 'exit', or 'q' to end.")
    print("Type 'draw' or 'd' to see a diagram of the graph.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() in ["draw", "d"]:
                print("Attempting to draw the graph...")
                draw_graph(graph)
                # Continue to the next loop to allow for more questions.
                continue
            stream_graph_updates(user_input)
        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C or end-of-file (Ctrl+D)
            print("\nGoodbye!")
            break
        except Exception as e:
            # Fallback for other errors, preserving original fallback logic
            print(f"\nAn unexpected error occurred: {e}")
            print("Switching to a non-interactive example.")
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    main()
