from ollama import chat
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage


class GuiAction(BaseModel):
    reasoning: Annotated[str, Field(description="describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task and decide if you need to take any action or not")]
    action: Annotated[Optional[Literal["press_key", "right_click", "double_click", "type", "wait", "FINISH"]], Field(description="Action to perform", default=None)]
    box_id: Annotated[Optional[int], Field(description="Bounding box ID", default=0)]
    x_coordinate: Annotated[Optional[int], Field(description="X coordinate to click provide if action is left_click, right_click or double_click", default=None)]
    y_coordinate: Annotated[Optional[int], Field(description="Y coordinate to click provide if action is left_click, right_click or double_click", default=None)]
    text: Annotated[Optional[str], Field(description="Text to type if action is type", default=None)]


def get_gui_action(
    image_path: str,
    elements: Optional[str] = None,
    previous_actions: Optional[List] = None,
    model: str = "qwen2.5vl:7b",
    screen_resolution: str = "1920x1080",
    system_prompt: Optional[str] = None,
    task_description: Optional[str] = None,
) -> GuiAction:
    try:
        # Build prompt
        user_input = ""
        previous_actions = ""

        for m in previous_actions:
            if isinstance(m, HumanMessage):
                user_input += m.content
            elif isinstance(m, AIMessage):
                previous_actions += m.content
        

        if system_prompt is None:
            system_prompt = """You are using a Windows device.
            You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
            You can only interact with the desktop GUI (no terminal or application menu access).

            You will be provided history of actions, screenshot of the UI and detected elements on the UI and consider your previos actions and the current screenshot to determine the next action.

            Your next action only include:
            - type: types a string of text.
            - double_click: move mouse to box id and double clicks to open any application.(strictly requires x_coordinate and y_coordinate)
            - right_click: move mouse to box id and right clicks.(strictly requires x_coordinate and y_coordinate)
            - double_click: move mouse to box id and double clicks.(strictly requires x_coordinate and y_coordinate)
            - wait: waits for 1 second for the device to load or respond.

            Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action.

            One Example:
            
            "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
            "Next Action": "left_click",
            

            Another Example:
            
            "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
            "Next Action": "type",
            "value": "Apple watch"
            

            IMPORTANT NOTES:
            1. If user asked to describe something on the UI just describe that do not include any action.
            2. If the actions history ony contains Humanmessage then it means it is the start of the actions history and you have to achieve that task.


            """

            user_prompt = user_prompt = f"""The user query is:
            {user_input} 
            This is your utmost importance that you have to achieve.

            The previous actions include(if the actions history is empty then it means it is the start of the actions history):
            {previous_actions}

            The current screen elements are:
            {elements}

            Your task is to analyze the previous actions and current screen elements and the screenshot containing bounding boxes on them with boxid provided and determine the next logical action.
            """

        # Call Ollama with structured output
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_path]
                }
            ],
            format=GuiAction.model_json_schema(),  # Enforce schema
        )

        # Validate against Pydantic model
        return GuiAction.model_validate_json(response.message.content)

    except Exception as e:
        print(f"Error getting GUI action: {e}")
        return GuiAction(
            reasoning=f"Error occurred: {str(e)}. Unable to analyze screenshot properly.",
            action="wait",
            box_id=0,
            x_coordinate=0,
            y_coordinate=0,
            text=""
        )


# Example usage
# if __name__ == "__main__":
#     action = get_gui_action(
#         image_path="parsed_screenshot.jpg",
#         task_description="describe the desktop background",
#     )

#     print(f"Action: {action.action}")
#     print(f"Reasoning: {action.reasoning}")
#     print(f"Coordinates: ({action.x_coordinate}, {action.y_coordinate})")
#     print(f"Box ID: {action.box_id}")
#     if action.text:
#         print(f"Text: {action.text}")
