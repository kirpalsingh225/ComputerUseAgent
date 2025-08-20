import gradio as gr
from computer_agent import build_graph
from langchain_core.messages import HumanMessage
import json

def agent_response_stream(user_input):
    graph = build_graph()
    state = {"messages": [HumanMessage(content=user_input)]}
    response = graph.invoke(state)
    output_lines = []
    image_path = "parsed_screenshot.jpg"
    for message in response["messages"]:
        reasoning = action = x = y = ""
        if hasattr(message, 'name') and message.name == "computer_agent":
            try:
                data = json.loads(message.content)
                if isinstance(data, str):
                    data = json.loads(data)
                reasoning = str(data.get("reasoning", ""))
                action = str(data.get("action", ""))
                x = str(data.get("x_coordinate", ""))
                y = str(data.get("y_coordinate", ""))
            except Exception as e:
                reasoning = f"Error parsing: {e}"
        else:
            reasoning = str(message.content) if hasattr(message, 'content') else str(message)
        output_lines.append(f"Reasoning: {reasoning}\nAction: {action}\nX: {x}\nY: {y}")
    final_output = "\n---\n".join(output_lines)
    return final_output, image_path

with gr.Blocks() as demo:
    gr.Markdown("# Computer Agent")
    user_input = gr.Textbox(label="Enter your command")
    submit_btn = gr.Button("Submit")
    output_box = gr.Textbox(label="Agent Output (Reasoning, Action, Coordinates)", lines=15, interactive=True, autofocus=True)
    image_box = gr.Image(label="Parsed Screenshot")
    submit_btn.click(
        agent_response_stream,
        inputs=user_input,
        outputs=[output_box, image_box]
    )

demo.launch()
