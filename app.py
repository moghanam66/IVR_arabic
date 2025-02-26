from flask import Flask, request, jsonify, send_from_directory, Response
import os
import asyncio
import openai
import pandas as pd
import ast
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import redis
import azure.cognitiveservices.speech as speechsdk
from rtclient import ResponseCreateMessage, RTLowLevelClient, ResponseCreateParams
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import nest_asyncio
nest_asyncio.apply()
 
# Import Bot Framework dependencies
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
 
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
 
# ------------------------------------------------------------------
# Configuration for Azure OpenAI, GPT‑4o realtime, Azure Search, Redis, Speech
# ------------------------------------------------------------------
 
# Azure OpenAI configuration for embeddings
OPENAI_API_KEY = "8929107a6a6b4f37b293a0fa0584ffc3"
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_ENDPOINT = "https://genral-openai.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Fast embedding model
 
# GPT‑4o realtime
RT_API_KEY = "9e76306d48fb4e6684e4094d217695ac"
RT_ENDPOINT = "https://general-openai02.openai.azure.com/"
RT_DEPLOYMENT = "gpt-4o-realtime-preview"
RT_API_VERSION = "2024-10-17"
 
# Azure Cognitive Search
SEARCH_SERVICE_NAME = "mainsearch01"          
SEARCH_INDEX_NAME = "id"                      
SEARCH_API_KEY = "Y6dbb3ljV5z33htXQEMR8ICM8nAHxOpNLwEPwKwKB9AzSeBtGPav"
 
# Redis
REDIS_HOST = "AiKr.redis.cache.windows.net"
REDIS_PORT = 6380
REDIS_PASSWORD = "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="
 
# Speech
SPEECH_KEY = "3c358ec45fdc4e6daeecb7a30002a9df"
SPEECH_REGION = "westus2"
 
# Thresholds for determining whether a search result is “good enough.”
SEMANTIC_THRESHOLD = 3.4
VECTOR_THRESHOLD = 0.91
 
# ------------------------------------------------------------------
# Initialize clients and load data
# ------------------------------------------------------------------
 
# Initialize the Azure OpenAI client (for embeddings)
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)
 
# Load Q&A data
try:
    qa_data = pd.read_csv("qa_data.csv", encoding="windows-1256")
    print("✅ CSV file loaded successfully!")
    print(qa_data.head())
except Exception as e:
    print(f"❌ Failed to load CSV file: {e}")
    exit()
 
# Normalize column names (convert to lowercase, trim spaces)
qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
 
# Convert the 'id' column to string (fix type conversion error)
if "id" in qa_data.columns:
    qa_data["id"] = qa_data["id"].astype(str)
 
# Verify required columns exist
if "question" not in qa_data.columns or "answer" not in qa_data.columns:
    print("❌ CSV file must contain 'question' and 'answer' columns.")
    exit()
 
 
 
# ------------------------------------------------------------------
# SEARCH & RESPONSE FUNCTIONS
# ------------------------------------------------------------------
 
 
   
# GPT‑4o REALTIME FALLBACK (ASYNC)
async def get_realtime_response(user_query):
    """
    Fallback function: Uses GPT‑4o realtime to generate an answer if both searches fail.
    Now with added instructions so that the model responds as an Egyptian man.
    """
    try:
        async with RTLowLevelClient(
            url=RT_ENDPOINT,
            azure_deployment=RT_DEPLOYMENT,
            key_credential=AzureKeyCredential(RT_API_KEY)
        ) as client_rt:
            # Prepend instruction for Egyptian persona
            instructions = "أنت رجل عربي. انا لا اريد ايضا اي bold points  في الاجابة  و لا اريد عنواين مرقمة" + user_query
            await client_rt.send(
                ResponseCreateMessage(
                    response=ResponseCreateParams(
                        modalities={"text"},
                        instructions=instructions
                    )
                )
            )
            done = False
            response_text = ""
            while not done:
                message = await client_rt.recv()
                if message is None:
                    print("❌ No message received from the real-time service.")
                    break
                if message.type == "response.done":
                    done = True
                elif message.type == "error":
                    done = True
                    print(f"Error: {message.error}")
                elif message.type == "response.text.delta":
                    response_text += message.delta
            return response_text
    except Exception as e:
        print(f"❌ Failed to get real-time response: {e}")
        return "عذرًا، حدث خطأ أثناء محاولة الاتصال بخدمة الدعم الفوري. يرجى المحاولة مرة أخرى لاحقًا."
 
 
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
 
# ------------------------------------------------------------------
# Bot Framework Integration
# ------------------------------------------------------------------
from botbuilder.schema import ConversationAccount
 
# Bot Framework credentials (set via environment or hard-code for testing)
MICROSOFT_APP_ID = "b0a29017-ea3f-4697-aef7-0cb05979d16c"
MICROSOFT_APP_PASSWORD = "2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ"
 
# Initialize Bot Framework adapter
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)
 
# Define a bot class that uses your get_response logic
class VoiceChatBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == "message":
            user_query = turn_context.activity.text
            print(f"Received message: {user_query}")
            response_text = await get_realtime_response(user_query)
            await turn_context.send_activity(response_text)
        elif turn_context.activity.type == "conversationUpdate":
            for member in turn_context.activity.members_added or []:
                if member.id != turn_context.activity.recipient.id:
                    welcome_message = "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
                    await turn_context.send_activity(welcome_message)
        else:
            await turn_context.send_activity(f"Received activity of type: {turn_context.activity.type}")
 
# Create an instance of the bot
bot = VoiceChatBot()
 
@app.route("/api/messages", methods=["POST"])
def messages():
    if request.headers.get("Content-Type", "") != "application/json":
        return Response("Invalid Content-Type", status=415)
   
    try:
        body = request.json
        print(body)
        if not body:
            print("❌ Empty request body received")
            return Response("Empty request body", status=400)
   
        print("🔍 Incoming request JSON:", json.dumps(body, indent=2, ensure_ascii=False))
   
        # Ensure the activity type is set
        if "type" not in body:
            body["type"] = "message"
            print("🔍 updated request JSON:", json.dumps(body, indent=2, ensure_ascii=False))
                   
        # Deserialize the incoming JSON into an Activity object
        activity = Activity().deserialize(body)
       
        if not activity.channel_id:
            activity.channel_id = body.get("channelId", "emulator")
        if not activity.service_url:
            activity.service_url = "https://linkdev-poc-cfb2fbaxbgf9d4dd.westeurope-01.azurewebsites.net"
       
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            auth_header = "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImltaTBZMnowZFlLeEJ0dEFxS19UdDVoWUJUayJ9.eyJhdWQiOiJiMGEyOTAxNy1lYTNmLTQ2OTctYWVmNy0wY2IwNTk3OWQxNmMiLCJpc3MiOiJodHRwczovL2xvZ2luLm1pY3Jvc29mdG9ubGluZS5jb20vZDZkNDk0MjAtZjM5Yi00ZGY3LWExZGMtZDU5YTkzNTg3MWRiL3YyLjAiLCJpYXQiOjE3NDA0NzY2MzcsIm5iZiI6MTc0MDQ3NjYzNywiZXhwIjoxNzQwNTYzMzM3LCJhaW8iOiJBU1FBMi84WkFBQUFzVnYrNXRUT0JkV3ZnVEYrQmVpUkd3azcyRTNKOXl6c1BjdHZjZmR5YU5ZPSIsImF6cCI6ImIwYTI5MDE3LWVhM2YtNDY5Ny1hZWY3LTBjYjA1OTc5ZDE2YyIsImF6cGFjciI6IjEiLCJyaCI6IjEuQVc0QUlKVFUxcHZ6OTAyaDNOV2FrMWh4MnhlUW9yQV82cGRHcnZjTXNGbDUwV3hlQVFCdUFBLiIsInRpZCI6ImQ2ZDQ5NDIwLWYzOWItNGRmNy1hMWRjLWQ1OWE5MzU4NzFkYiIsInV0aSI6IkhxVi1ZcHFoalVtZlJmXzlOXzhuQUEiLCJ2ZXIiOiIyLjAifQ.tkkP-QoPHHc4PqiUJNVUW-VsQwkhHmbFbbf_ZPviliEI7ldAmSYNbEbde9JsZwSHzFNsrYm_Ke3keSa_CVuRshFV2xXoMHTJtDdrU5NyfvN0ifIR1eUoLjIWMUDt0mDNXpHUjvBXKSbO-H7vejz3pk8xTejOMSR36iT6jpxPBEVH-5UdonJPAWGFHjouisOgfginuMJa4ZAFFeivdnGyubw67K8tEJejgwkFllevYaVDM5NEPTZMpDFFhwQKrPZQw_8spE1XEA_LK-SdrzIyWPO1rHbcDkKP5lhD2bHZHBKtrWiZzR_n1D7gZZ0AdT_bHDmJI26NplBEw7F9wNstoA"
 
        print("auth: ", auth_header)
   
        async def call_bot():
            await adapter.process_activity(activity, auth_header, bot.on_turn)
   
        # Create a new event loop for this request to avoid using a closed loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(call_bot())
        loop.close()
        return Response(status=201)
   
    except Exception as e:
        print(f"❌ Error in /api/messages: {e}")
        return Response("Internal server error", status=500)
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
