from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaLLM
from langgraph.graph.message import MessagesState
import ollama
from test import process_image
import pyautogui
import json
from langchain_core.messages import HumanMessage, AIMessage
import time
# from local_qwen import get_gui_action
# from claude_code import get_gui_action
from lc import get_gui_action
from langsmith import traceable
from dotenv import load_dotenv
import logging
import os

load_dotenv()



@traceable
def computer_node(state: MessagesState):

    pyautogui.screenshot("screenshot.png")

    elements = process_image(
        image_path=r"screenshot.png",
        yolo_model_path='weights/icon_detect/model.pt',
        caption_model_name="florence2",
        caption_model_path="weights/icon_caption_florence",
        device='cuda',
        box_threshold=0.05,
        iou_threshold=0.7,
        text_threshold=0.9,
        use_paddleocr=True,
        batch_size=32
    )

    logging.info(f"Elements detected: {elements}")

    print(elements[:50])

    key = "" #os.getenv("OPENROUTER_API_KEY")

    response = get_gui_action(api_key=key, image_path="screenshot.png",elements=elements, previous_actions=state["messages"])

    # response = get_gui_action(image_path="screenshot.png", elements=elements, previous_actions=state["messages"])
    logging.info(f"Response: {response}")
   

    return {
        "messages": [AIMessage(content=json.dumps(response.model_dump_json()), name="computer_agent")]
    }

@traceable
def execute_action(state: MessagesState):
    last_message = state["messages"][-1]  
    try:
        action_data = json.loads(last_message.content)

        if isinstance(action_data, str):
            action_data = json.loads(action_data)

    except json.JSONDecodeError as e:
        return {"messages": [f"Failed to decode action data: {e}"]}

    action = action_data.get("action")

    if action == "press_key":
        key_value = action_data.get("key_value")
        if key_value in ["enter", "space"]:
            if key_value == "enter":
                pyautogui.press("enter")
                time.sleep(3)
            elif key_value == "space":
                pyautogui.press("space")
                time.sleep(3)
            logging.info(f"Pressed key: {key_value}")
            return {"messages": [AIMessage(content=f"Pressed key: {key_value}")]}

    elif action == "left_click":
        x, y = action_data.get("x_coordinate"), action_data.get("y_coordinate")
        if x is not None and y is not None:
            pyautogui.click(x, y)
            time.sleep(3)
            logging.info(f"Performed left click at ({x}, {y})")
            return {"messages": [AIMessage(content=f"Performed left click at ({x}, {y})")]}
        else:
            print("Left click action requires x and y coordinates.")

    elif action == "right_click":
        x, y = action_data.get("x_coordinate"), action_data.get("y_coordinate")
        if x is not None and y is not None:
            pyautogui.rightClick(x, y)
            time.sleep(3)
            logging.info(f"Performed right click at ({x}, {y})")
            return {"messages": [AIMessage(content=f"Performed right click at ({x}, {y})")]}

    elif action == "type":
        text = action_data.get("value")
        if text:
            pyautogui.typewrite(text)
            time.sleep(3)
            logging.info(f"Typed text: {text}")
            return {"messages": [AIMessage(content=f"Typed text: {text}")]}
        
    elif action == "double_click":
        x, y = action_data.get("x_coordinate"), action_data.get("y_coordinate")
        if x is not None and y is not None:
            pyautogui.doubleClick(x, y)
            time.sleep(3)
            logging.info(f"Performed double click at ({x}, {y})")
            return {"messages": [AIMessage(content=f"Performed double click at ({x}, {y})")]}

    elif action == "wait":
        time.sleep(5)
        logging.info("Waited for 5 seconds.")
        return {"messages": [AIMessage(content="Waited for 5 seconds.")]}
    
    elif action == "scroll":
        pyautogui.scroll(-500)  # Scroll up by 100 units
        time.sleep(3)
        logging.info("Scrolled up by 100 units.")
        return {"messages": [AIMessage(content="Scrolled down by 100 units.")]}

    return {"messages": [AIMessage(content=f"Unknown action: {action}")]}
    


@traceable
def router(state: MessagesState):
    last_message = state["messages"][-1] if state["messages"] else ""
    outer = json.loads(last_message.content)

    if isinstance(outer, str):
        inner = json.loads(outer)
    else:
        inner = outer

    logging.info(f"Router inner: {inner.get('action')}")

    if inner.get("action") == "FINISH":
        return "end"
    
    return "execution"


def build_graph():
    builder = StateGraph(MessagesState)


    builder.add_node("computer_node", computer_node)
    builder.add_node("execute_action", execute_action)


    builder.add_edge(START, "computer_node")
    builder.add_conditional_edges("computer_node", router, {"end": END, "execution": "execute_action"})
    builder.add_edge("execute_action", "computer_node")

    return builder.compile()



if __name__ == "__main__":
    log_path = r""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Force logging config (disable existing handlers)
    logging.basicConfig(
        filename=log_path,
        filemode="w",  # overwrite each run; use "a" to append
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,  # <- this makes sure config is applied even if something else set it before
    )

    logging.info("Logging system initialized")
    graph = build_graph()
    
    response = graph.invoke(
        {
            "messages": [HumanMessage(content="you will be given a vs code environment and in line 415 there is text Content you have to add one more t to it to make it Contentt")],
        }
    )

    for message in response["messages"]:
        message.pretty_print()

    logging.info("Agent log test: script ended")
    logging.shutdown()




