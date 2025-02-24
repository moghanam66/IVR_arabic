import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops
 
from flask import Flask, request, jsonify, send_from_directory, Response
import os
import asyncio
import openai
import pandas as pd
import ast
import json
import time
import numpy as np
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import redis
import azure.cognitiveservices.speech as speechsdk
from rtclient import ResponseCreateMessage, RTLowLevelClient, ResponseCreateParams
from flask_cors import CORS
 
# Bot Framework dependencies
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
 
# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)
 
# ------------------------------------------------------------------
# Configuration for Azure OpenAI, GPT‑4o realtime, Azure Search, Redis, Speech
# ------------------------------------------------------------------
 
# Azure OpenAI configuration for embeddings
OPENAI_API_KEY = "8929107a6a6b4f37b293a0fa0584ffc3"
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_ENDPOINT = "https://genral-openai.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Fast embedding model
 
# GPT‑4o realtime configuration
RT_API_KEY = "9e76306d48fb4e6684e4094d217695ac"
RT_ENDPOINT = "https://general-openai02.openai.azure.com/"
RT_DEPLOYMENT = "gpt-4o-realtime-preview"
RT_API_VERSION = "2024-10-17"
 
# Azure Cognitive Search configuration
SEARCH_SERVICE_NAME = "mainsearch01"          
SEARCH_INDEX_NAME = "id"                      
SEARCH_API_KEY = "Y6dbb3ljV5z33htXQEMR8ICM8nAHxOpNLwEPwKwKB9AzSeBtGPav"
 
# Redis configuration
REDIS_HOST = "AiKr.redis.cache.windows.net"
REDIS_PORT = 6380
REDIS_PASSWORD = "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="
 
# Speech configuration
SPEECH_KEY = "3c358ec45fdc4e6daeecb7a30002a9df"
SPEECH_REGION = "westus2"
 
# Thresholds for search matching
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
 
# Normalize column names
qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
if "id" in qa_data.columns:
    qa_data["id"] = qa_data["id"].astype(str)
if "question" not in qa_data.columns or "answer" not in qa_data.columns:
    print("❌ CSV file must contain 'question' and 'answer' columns.")
    exit()
 
# EMBEDDING GENERATION
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        print(f"✅ Embedding generated for text: {text}")
        return embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding for text '{text}': {e}")
        return None
 
if "embedding" not in qa_data.columns or qa_data["embedding"].isnull().all():
    qa_data["embedding"] = qa_data["question"].apply(get_embedding)
    qa_data.to_csv("embedded_qa_data.csv", index=False)
    print("✅ Embeddings generated and saved to 'embedded_qa_data.csv'.")
else:
    def convert_embedding(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                print("❌ Failed to parse embedding:", e)
                return None
        return x
    qa_data["embedding"] = qa_data["embedding"].apply(convert_embedding)
    print("✅ Using existing embeddings from CSV.")
 
# Normalize question text
qa_data["question"] = qa_data["question"].str.strip().str.lower()
 
# UPLOAD DOCUMENTS TO AZURE COGNITIVE SEARCH
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net/",
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)
 
documents = qa_data.to_dict(orient="records")
try:
    upload_result = search_client.upload_documents(documents=documents)
    print(f"✅ Uploaded {len(documents)} documents to Azure Search. Upload result: {upload_result}")
except Exception as e:
    print(f"❌ Failed to upload documents: {e}")
 
# Simple query for verification
try:
    simple_results = search_client.search(
        search_text="*",
        select=["question", "answer"],
        top=3
    )
    print("Simple query results:")
    for doc in simple_results:
        print(doc)
except Exception as e:
    print(f"❌ Simple query error: {e}")
 
# INITIALIZE REDIS CLIENT
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        ssl=True,
        decode_responses=True,
        password=REDIS_PASSWORD
    )
    redis_client.ping()
    print("✅ Successfully connected to Redis!")
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
 
# ------------------------------------------------------------------
# SEARCH & RESPONSE FUNCTIONS
# ------------------------------------------------------------------
 
def check_redis_cache(query):
    try:
        cached_answer = redis_client.get(query)
        if cached_answer:
            print(f"✅ Using cached answer for query: {query}")
            return cached_answer
    except Exception as e:
        print(f"❌ Redis error: {e}")
    return None
 
def get_best_match(query):
    cached_response = check_redis_cache(query)
    if cached_response:
        return cached_response
 
    # Semantic Search
    try:
        semantic_results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config-default",
            query_caption="extractive",
            select=["question", "answer"],
            top=3
        )
        semantic_answer = next(semantic_results, None)
        if semantic_answer:
            reranker_score = semantic_answer["@search.reranker_score"]
            if reranker_score is not None and reranker_score >= SEMANTIC_THRESHOLD:
                answer = semantic_answer["answer"]
                print("✅ Found match using Semantic Search with score", reranker_score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Semantic search result score below threshold:", reranker_score)
        else:
            print("❌ No semantic search answers found.")
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
 
    # Vector Search
    try:
        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding
 
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=50,
            fields="embedding"
        )
        vector_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["question", "answer"],
            top=3
        )
        best_vector = next(vector_results, None)
        if best_vector:
            score = best_vector.get("@search.score", 0)
            if score >= VECTOR_THRESHOLD:
                answer = best_vector["answer"]
                print("✅ Found match using Vector Search with score", score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Vector search result score below threshold:", score)
        else:
            print("❌ No vector search results found.")
    except Exception as e:
        print(f"❌ Vector search error: {e}")
 
    print("❌ No match found using Semantic or Vector Search")
    return None
 
async def get_realtime_response(user_query):
    try:
        async with RTLowLevelClient(
            url=RT_ENDPOINT,
            azure_deployment=RT_DEPLOYMENT,
            key_credential=AzureKeyCredential(RT_API_KEY)
        ) as client_rt:
            # Prepend instructions for Egyptian persona
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
 
async def get_response(user_query):
    print(f"🔍 Processing query: {user_query}")
    response = get_best_match(user_query)
    if response:
        print(f"✅ Found response in cache or search: {response}")
        return response
 
    print("🔍 No match found, falling back to GPT‑4o realtime...")
    realtime_response = await get_realtime_response(user_query)
    if realtime_response:
        print(f"✅ GPT‑4o realtime response: {realtime_response}")
        try:
            redis_client.set(user_query, realtime_response, ex=3600)
            print("✅ Response cached in Redis.")
        except Exception as e:
            print(f"❌ Failed to cache response in Redis: {e}")
        return realtime_response
    else:
        return "عذرًا، لم أتمكن من العثور على إجابة. يرجى المحاولة مرة أخرى لاحقًا."
 
# ------------------------------------------------------------------
# SPEECH RECOGNITION & SYNTHESIS SETUP
# ------------------------------------------------------------------
 
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "ar-EG"
speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"
 
def recognize_speech():
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print("Listening... (Speak in Egyptian Arabic)")
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"You said: {result.text}")
        return result.text
    else:
        print(f"Speech not recognized: {result.reason}")
        return ""
 
def speak_response(text):
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Speech synthesis canceled:")
        print("  Reason: {}".format(cancellation.reason))
        print("  Error Details: {}".format(cancellation.error_details))
 
def clean_text(text):
    return text.strip(" .،!؛؟").lower()
 
def detect_critical_issue(text):
    trigger_sentences = [
        "تم اكتشاف اختراق أمني كبير.",
        "تمكن قراصنة من الوصول إلى بيانات حساسة.",
        "هناك هجوم إلكتروني على النظام الخاص بنا.",
        "تم تسريب بيانات المستخدمين إلى الإنترنت.",
        "رصدنا محاولة تصيد إلكتروني ضد موظفينا.",
        "تم استغلال ثغرة أمنية في الشبكة.",
        "هناك محاولة وصول غير مصرح بها إلى الملفات السرية."
    ]
    # Get embeddings for trigger sentences
    trigger_embeddings = np.array([get_embedding(sent) for sent in trigger_sentences])
    text_embedding = np.array(get_embedding(text)).reshape(1, -1)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(text_embedding, trigger_embeddings)
    max_similarity = np.max(similarities)
    if max_similarity > 0.9:
        print("This issue should be passed to a human.")
        return True
    return False
 
# ------------------------------------------------------------------
# ASYNCHRONOUS VOICE CHAT LOOP & ROUTES
# ------------------------------------------------------------------
 
async def voice_chat_loop():
    print("🤖 Arabic Voice Bot Ready! Say 'إنهاء' or 'خروج' to exit.")
    while True:
        user_query = recognize_speech()
        if clean_text(user_query) in ["إنهاء", "خروج"]:
            print("👋 Goodbye!")
            speak_response("مع السلامة!")
            break
 
        if detect_critical_issue(user_query):
            response = "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
            print(f"🤖 Bot: {response}")
            speak_response(response)
            continue
 
        response = await get_response(user_query)
        print(f"🤖 Bot: {response}")
        speak_response(response)
 
async def voice_chat(user_query):
    try:
        if not user_query:
            return "في انتظار اوامرك"
        if clean_text(user_query) in ["إنهاء", "خروج"]:
            print("👋 Goodbye!")
            return "مع السلامة"
        if detect_critical_issue(user_query):
            response = "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
            return response
        response = await get_response(user_query)
        return response
    except Exception as e:
        print(f"Error in /voice-chat: {e}")
        return "عذرًا، حدث خطأ أثناء معالجة استفسارك."
 
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
 
# ------------------------------------------------------------------
# Bot Framework Integration
# ------------------------------------------------------------------
 
MICROSOFT_APP_ID = "b0a29017-ea3f-4697-aef7-0cb05979d16c"
MICROSOFT_APP_PASSWORD = "2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ"
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)
 
class VoiceChatBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == "message":
            user_query = turn_context.activity.text
            print(f"Received message via Bot Framework: {user_query}")
            response_text = await voice_chat(user_query)
            await turn_context.send_activity(response_text)
        elif turn_context.activity.type == "conversationUpdate":
            for member in turn_context.activity.members_added or []:
                if member.id != turn_context.activity.recipient.id:
                    welcome_message = "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
                    await turn_context.send_activity(welcome_message)
        else:
            await turn_context.send_activity(f"Received activity of type: {turn_context.activity.type}")
 
bot = VoiceChatBot()
 
@app.route("/api/messages", methods=["POST"])
def messages():
    if request.headers.get("Content-Type", "") != "application/json":
        return Response("Invalid Content-Type", status=415)
   
    try:
        body = request.json
        if not body:
            print("❌ Empty request body received")
            return Response("Empty request body", status=400)
   
        print("🔍 Incoming request JSON:", json.dumps(body, indent=2, ensure_ascii=False))
   
        if "type" not in body:
            print("❌ Missing activity type")
            return Response("Missing activity type", status=400)
   
        # Deserialize the activity and ensure required fields are set
        activity = Activity().deserialize(body)
        if not activity.channel_id:
            activity.channel_id = body.get("channelId", "emulator")
   
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            auth_header = "eyJ0eXAiOiJKV1QiLCJub25jZSI6Im5XV2RDRXBEVDlfWUNUNDdZeVdlSlpGZG83eGxzdVRWMi0wcl9EeUdkQ00iLCJhbGciOiJSUzI1NiIsIng1dCI6ImltaTBZMnowZFlLeEJ0dEFxS19UdDVoWUJUayIsImtpZCI6ImltaTBZMnowZFlLeEJ0dEFxS19UdDVoWUJUayJ9.eyJhdWQiOiJodHRwczovL2dyYXBoLm1pY3Jvc29mdC5jb20vIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvMTM4ZTJmYmEtODBkYS00OTEzLTk1MjQtNzg4NDIzMjRlMDc4LyIsImlhdCI6MTc0MDQwMDc2NCwibmJmIjoxNzQwNDAwNzY0LCJleHAiOjE3NDA0MDYyMjEsImFjY3QiOjEsImFjciI6IjEiLCJhaW8iOiJBWlFBYS84WkFBQUFqSzZldTRJQ25oQ3NDaUd6T3h0bnRFQkRoODdXaXdNWE5mdjBTSlVCV1c2dUpOV1pJVmVsc2VIaTZ4VHR6Y2FIK21vL1gzVng3ajFodnNsMFJxOE9hQUczOVFsSzErZlFpYlhtSlNCbzdWeU1MNWtFa1ZRZmpQS3MrV2NGUnJvQldlQ1FmenN5WWU0L0hhcUpPM3FjZ1lpNE1qandZWW9kNk9wdjJnL0crcE9KaytHd1MvZ1oxSFU4N3kvR0xKTEoiLCJhbHRzZWNpZCI6IjU6OjEwMDMyMDA0MEZCNjdGOTAiLCJhbXIiOlsicHdkIl0sImFwcF9kaXNwbGF5bmFtZSI6Iml2cl9hcmFiaWNfbGluayIsImFwcGlkIjoiYzU0ODQ0MDItYWY3Yy00ZTViLTg3NjYtZDk5NjY5MDE3YmM0IiwiYXBwaWRhY3IiOiIxIiwiZW1haWwiOiJtb2hhbWVkLmdoYW5hbUBsaW5rZGV2LmNvbSIsImlkcCI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0L2IwODQwMzFiLWNkYjEtNDIyZS1hNzZhLWE2NTEwY2RiMTdmOC8iLCJpZHR5cCI6InVzZXIiLCJpcGFkZHIiOiIxOTcuNTQuMjYuMjIwIiwibmFtZSI6Ik1vaGFtZWQgR2hhbmFtIiwib2lkIjoiZTNhN2M4ODctNTk1Yy00ZDYyLTk1OTItNzdmYTY2NmEwNzA2IiwicGxhdGYiOiIzIiwicHVpZCI6IjEwMDMyMDA0NTNEQTYyRTYiLCJyaCI6IjEuQVY0QXVpLU9FOXFBRTBtVkpIaUVJeVRnZUFNQUFBQUFBQUFBd0FBQUFBQUFBQUFSQVhwZUFBLiIsInNjcCI6IlVzZXIuUmVhZCIsInNpZCI6IjMxOTdlYzkzLTk2NDgtNGZiOS1iNzM1LWY3YTFiODEwNDczYSIsInNpZ25pbl9zdGF0ZSI6WyJrbXNpIl0sInN1YiI6IjZLOTRsUXBObGkzU1g4ZERsZWNySFR1RDJHc3BFSFEwQXlYY2xWZFFoZ2ciLCJ0ZW5hbnRfcmVnaW9uX3Njb3BlIjoiRVUiLCJ0aWQiOiIxMzhlMmZiYS04MGRhLTQ5MTMtOTUyNC03ODg0MjMyNGUwNzgiLCJ1bmlxdWVfbmFtZSI6Im1vaGFtZWQuZ2hhbmFtQGxpbmtkZXYuY29tIiwidXRpIjoicU5JRzdWREoxME9ydVg1b05XQ2FBQSIsInZlciI6IjEuMCIsIndpZHMiOlsiY2YxYzM4ZTUtMzYyMS00MDA0LWE3Y2ItODc5NjI0ZGNlZDdjIiwiMTNiZDFjNzItNmY0YS00ZGNmLTk4NWYtMThkM2I4MGYyMDhhIl0sInhtc19pZHJlbCI6IjQgNSIsInhtc190Y2R0IjoxNTMyNjEyMjE2fQ.kRTPVdwqttCtOlFdjURBYkjQuhqPVrRf5k_zzRi7Sg1N2C-sbasq4MYf_FK9ddVKRWN_a-vVXyQVvnbbE94yl8IjrlTqTy9Klz4BZOasnQlKQRnNEdVKvrk5z341Jah0aV3dSrlcSSdr46NIwcuHeiwy1TrDm5ZMQX4uUnm91WJN0oIdojc2Q53ZD2l5fVvv63bBlrc8Vs2FfTceMX7BWsTrIAeyyJhK9lwDMQ1ti2esFp7CJSR4Lr8MhCWJK9nRYg3KPqFsClofLg0w3TMHRISL7TkwCFNg8G8RA8lURHMjC-6PNim12fmkwrjVOtHx7WOcWLT6Mgyq2g3KdXqU7Q"
   
        async def call_bot():
            await adapter.process_activity(activity, auth_header, bot.on_turn)
   
        loop = asyncio.get_event_loop()
        loop.run_until_complete(call_bot())
        return Response(status=201)
   
    except Exception as e:
        print(f"❌ Error in /api/messages: {e}")
        return Response("Internal server error", status=500)
 
# ------------------------------------------------------------------
# Run the Flask application
# ------------------------------------------------------------------
 
if __name__ == "__main__":
    app.run(debug=True)
