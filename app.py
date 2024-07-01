import gradio as gr
from src.chat_agent import ChatAgent

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#! Should read data  once and then use it in the chat_agent

app = FastAPI()
chat = ChatAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/")
async def chat_endpoint(input_data: dict):
    question = input_data.get("text")
    if not question:
        return {"error": "No question provided"}
    response = chat.query(question)
    return {"response": response}

async def gradio_chat(text):
    response = chat.query(text)
    return response

iface = gr.Interface(fn=gradio_chat, inputs="text", outputs="text")
iface.launch()
