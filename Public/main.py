from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re
import logging
from typing import Optional, List, Dict
from datetime import datetime
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Security Configurations
class SecurityConfig:
    MAX_MESSAGE_LENGTH = 2500
    BLOCKED_KEYWORDS = {'sql', 'exec', 'eval', 'delete', 'drop table'}
    SENSITIVE_TOPICS = {
        'explicit_content': ['porn', 'nsfw'],
        'hate_speech': ['hate', 'racist'],
        'violence': ['kill', 'attack'],
    }

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=SecurityConfig.MAX_MESSAGE_LENGTH)
    context: Optional[str] = Field(None, max_length=500)

    @validator("message")
    def validate_message_content(cls, v):
        lower_message = v.lower()
        for keyword in SecurityConfig.BLOCKED_KEYWORDS:
            if keyword in lower_message:
                raise ValueError("Message contains prohibited content")
        for _, terms in SecurityConfig.SENSITIVE_TOPICS.items():
            if any(term in lower_message for term in terms):
                raise ValueError("Message contains inappropriate content")
        return v

class ResponseFilter:
    @staticmethod
    def filter_output(response: str) -> str:
        """Trims response length and filters restricted patterns"""
        return response[:4096] if len(response) > 4096 else response

class PromptEngineering:
    SYSTEM_PROMPT = """You are a helpful AI assistant. Follow these rules:
    1. Do not generate harmful or explicit content.
    2. Provide factual and helpful information only.
    """

    @staticmethod
    def create_safe_prompt(user_message: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Creates a structured prompt for the AI model"""
        messages = [{"role": "system", "content": PromptEngineering.SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})
        return messages

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware (Allows all origins; restrict in production)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize Hugging Face API Client
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Add it to your environment variables.")

client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    token=HF_TOKEN
)

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """Handles chat requests and sends them to the AI model"""
    try:
        logger.info(f"Request received: {chat_message.message[:100]}...")

        messages = PromptEngineering.create_safe_prompt(chat_message.message, chat_message.context)
        
        # Ensure valid API call
        response = client.text_generation(messages[-1]["content"], max_new_tokens=2048, temperature=0.7)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate a response.")

        filtered_response = ResponseFilter.filter_output(response)
        
        logger.info("Response generated successfully.")
        return {"response": filtered_response}

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# Use Mangum to deploy on Firebase
handler = Mangum(app)
