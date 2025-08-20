from openai import OpenAI
import base64
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage
import logging 

class GuiAction(BaseModel):
    reasoning: Annotated[str, Field(description="Your reasoning about the next action to take or to handle tasks that doesnt require any action")]
    action: Annotated[Optional[Literal["left_click", "right_click", "press_key", "scroll", "type", "wait", "FINISH"]], Field(description="Action to perform", default=None)]
    box_id: Annotated[Optional[int], Field(description="Bounding box ID", default=0)]
    x_coordinate: Annotated[Optional[int], Field(description="X coordinate to move the mouse to (if action is mouse_move, left_click or right_click)", default=None)]
    y_coordinate: Annotated[Optional[int], Field(description="Y coordinate to move the mouse to (if action is mouse_move, left_click or right_click)", default=None)]
    value: Annotated[Optional[str], Field(description="Text to type if action is type", default=None)]
    key_value: Annotated[Optional[Literal["enter", "space"]], Field(description="Key to press if action is press_key", default=None)]

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

logger = logging.getLogger(__name__)

def get_gui_action(
    image_path: str,
    api_key: str,
    elements: Optional[str] = None,
    previous_actions: Optional[List] = None,
    model: str = "qwen/qwen2.5-vl-72b-instruct:free",
    screen_resolution: str = "1920x1080",
    base_url: str = "https://openrouter.ai/api/v1",
    system_prompt: Optional[str] = None,
) -> GuiAction:
    """
    Get structured GUI action from OpenAI API
    
    Args:
        image_path: Path to the screenshot image
        task_description: Description of what you want to accomplish
        api_key: OpenRouter API key
        elements: List of detected elements with descriptions (optional)
        previous_actions: List of previous actions taken (optional)
        model: Model to use for inference
        screen_resolution: Screen resolution string
        base_url: API base URL
        system_prompt: Custom system prompt (optional)
        
    Returns:
        GuiAction object with structured response
    """
    # Initialize client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    

    user_input = ""
    previous_actions_text = ""

    logger.info(f"Using model: {model}")

    # Fix: Handle None case and extract content correctly
    if previous_actions:
        for m in previous_actions:
            if isinstance(m, HumanMessage):
                user_input += m.content + "\n"
            elif isinstance(m, AIMessage):
                previous_actions_text += m.content + "\n"

    # Encode image
    base64_image = encode_image(image_path)

    # Build the prompt
    user_prompt = f"""The user query/task is:
    {user_input}
    This is your utmost importance that you have to achieve. Always check whether the user task is achieved or not and if its achieved then set action to FINISH.

    The previous actions include (if empty, this is the start):
    {previous_actions}

    The current screen elements with their center coordinates are provided and contains Box ID of element, content contains name of the element and Center Coords contains the center coordinates of the element use from the elements list and strictly use the center coordinates from these elements and strictly dont assume any coordinates:
    {elements if elements else "No elements provided"}

    Analyse the screenshot under images.

    Your task is to analyze the previous actions and current screen elements and the screenshot being provided to you and take the next logical action to achieve the task.

    """
    try:
        # Use custom system prompt or default one
        if system_prompt is None:
            system_prompt = """You are a GUI automation agent that analyzes UI screenshots with bounding boxes and labeled IDs. 
            You will be provided with a screenshot of the UI with bounding boxes with Box ID's on them and also the elements on the screen with their name and center coordinates to interact with them.
            
            Your task is to reason with the given UI screenshot and the previous actions taken to provide the best next action to do.
            
            Available actions:
            2. left_click: Perform a left mouse click at the current mouse position
            3. right_click: Perform a right mouse click at the current mouse position  
            4. type: Type a specific text at the current mouse position
            5. wait: Wait for UI to update. After every action.
            6. FINISH: Task completed
            
            Always provide valid x_coordinate and y_coordinate within the screen bounds and select appropriate box_id based on bounding boxes visible in the image.
            Include a detailed annotated description of what you see in the reasoning."""
        # Make API call
        completion = client.beta.chat.completions.parse(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "GUI Agent",
            },
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format=GuiAction
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        # Fallback action in case of error
        print(f"Error getting GUI action: {e}")
        return GuiAction(
            reasoning=f"Error occurred: {str(e)}. Unable to analyze the screenshot properly.",
            action="wait",
            box_id=0,
            x_coordinate=0,
            y_coordinate=0,
            text=""
        )

# Example usage
# if __name__ == "__main__":
#     # Your API key
#     API_KEY = ""
    
#     # Get action with custom system prompt
#     custom_system_prompt = """
#     You are a GUI agent that analyzes UI screenshots with bounding boxes and labeled IDs. You are working on a windows machine.
#     You will be provided with a screenshot of the UI with bounding boxes with Box ID's on them and also the elements on the screen with their name and center coordinates to interact with them.
    
#     Your task is to reason with the given UI screenshot and the previous actions taken to provide the best next action to do.
    
#     You can do following actions:
#     2. left_click: Perform a left mouse click at the current mouse position.
#     3. right_click: Perform a right mouse click at the current mouse position.
#     4. type_text: Type a specific text at the current mouse position.
#     5. wait: Wait for UI to update.
#     6. FINISH: Task completed.

#     Always use coordinates from the provided elements and select the appropriate box_id based on bounding boxes visible in the image.
#     """
    
#     action = get_gui_action(
#         image_path="parsed_screenshot.jpg",
#         api_key=API_KEY,
#         screen_resolution="1920x1080",
#         system_prompt=custom_system_prompt
#     )
    
#     # Use the result
#     print(f"Action: {action.action}")
#     print(f"Reasoning: {action.reasoning}")
#     print(f"Coordinates: ({action.x_coordinate}, {action.y_coordinate})")
#     print(f"Box ID: {action.box_id}")
#     if action.value:
#         print(f"Text: {action.value}")