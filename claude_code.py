from ollama import chat
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import base64


class GuiAction(BaseModel):
    reasoning: Annotated[str, Field(description="describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task and decide if you need to take any action or not")]
    action: Annotated[Optional[Literal["press_key", "right_click", "left_click", "type", "wait", "FINISH"]], Field(description="Action to perform.", default=None)]
    box_id: Annotated[Optional[int], Field(description="Bounding box ID", default=0)]
    x_coordinate: Annotated[Optional[int], Field(description="X coordinate to click provide if action is left_click, right_click or double_click", default=None)]
    y_coordinate: Annotated[Optional[int], Field(description="Y coordinate to click provide if action is left_click, right_click or double_click", default=None)]
    value: Annotated[Optional[str], Field(description="Text to type if action is type", default=None)]


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for Ollama API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
        # Build prompt - FIXED LOGIC
        user_input = ""
        previous_actions_text = ""

        # Fix: Handle None case and extract content correctly
        if previous_actions:
            for m in previous_actions:
                if isinstance(m, HumanMessage):
                    user_input += m.content + "\n"
                elif isinstance(m, AIMessage):
                    previous_actions_text += m.content + "\n"
        
        # If no user input from messages and task_description provided, use task_description
        if not user_input and task_description:
            user_input = task_description

        if system_prompt is None:
            system_prompt = """You are using a Windows device.
            You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
            You can only interact with the desktop GUI (no terminal or application menu access).

            You will be provided history of actions, screenshot of the UI and detected elements on the UI and consider your previous actions and the current screenshot to determine the next action.

            Your next action can only include:
            - type: types a string of text.
            - left_click: move mouse to coordinates and left clicks (requires x_coordinate and y_coordinate)
            - right_click: move mouse to coordinates and right clicks (requires x_coordinate and y_coordinate)
            - wait: waits for 1 second for the device to load or respond.
            - FINISH: when the task is completed.

            Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action.

            IMPORTANT NOTES:
            1. If user asked to describe something on the UI, just describe that and set action to null.
            2. If the actions history only contains HumanMessage then it means it is the start of the actions history and you have to achieve that task.
            3. Always provide reasoning for your decision.
            4. For click actions, always provide x_coordinate and y_coordinate.
            """

        user_prompt = f"""The user query/task is:
        {user_input} 
        This is your utmost importance that you have to achieve. Always check whether the user task is achieved or not and if its achieved then set action to FINISH.

        The previous actions include (if empty, this is the start):
        {previous_actions_text}

        The current screen elements are:
        {elements if elements else "No elements provided"}

        Analyse the screenshot under images.

        Your task is to analyze the previous actions and current screen elements and the screenshot being provided to you and take the next logical action to achieve the task.

        """

        with open("user_prompt.txt", "w") as f:
            f.write(user_prompt)


        # FIXED: Encode image to base64
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [encode_image_to_base64(image_path)]  
                }
            ],
            format=GuiAction.model_json_schema(),
        )

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
#         image_path=r"",
#         task_description="describe the desktop background",
#         previous_actions=[HumanMessage(content="What is on my desktop?")]
#     )

#     print(f"Action: {action.action}")
#     print(f"Reasoning: {action.reasoning}")
#     print(f"Coordinates: ({action.x_coordinate}, {action.y_coordinate})")
#     print(f"Box ID: {action.box_id}")
#     if action.value:
#         print(f"Text: {action.value}")