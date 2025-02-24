from flask import Flask, request, jsonify
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
from flask import Flask, request, jsonify, send_from_directory
import asyncio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes





# Azure OpenAI configuration for embeddings
OPENAI_API_KEY = "9e76306d48fb4e6684e4094d217695ac"
OPENAI_API_VERSION = "2024-10-01-preview"
OPENAI_ENDPOINT = "https://general-openai02.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Fast embedding model

# GPT‑4o realtime 
RT_API_KEY = "FZPood5bYLJ1peRf3cYHR29GVIIBrgryH5TikPUnjOemGzULmkY5JQQJ99BBACHYHv6XJ3w3AAABACOGoYWK"
RT_ENDPOINT = "https://gptkr.openai.azure.com/openai/realtime/"
RT_DEPLOYMENT = "gpt-4o-realtime-preview"
RT_API_VERSION = "2024-12-17"

# Azure Cognitive Search 
SEARCH_SERVICE_NAME = "search-link"          
SEARCH_INDEX_NAME = "id"                      
SEARCH_API_KEY = "OGIjhICbQnq4aycCnVQi29stqrLQl4lLfa4ClEYEKuAzSeBoGsjN"

# Redis 
REDIS_HOST = "AiKr.redis.cache.windows.net"
REDIS_PORT = 6380
REDIS_PASSWORD = "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="

# Speech 
SPEECH_KEY = "96f3229ebe7f4fe9ae4e3a1d01bf2184"
SPEECH_REGION = "eastus2"

# Thresholds for determining whether a search result is “good enough.”
SEMANTIC_THRESHOLD = 3.4 
VECTOR_THRESHOLD = 0.91 

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

# EMBEDDING GENERATION
def get_embedding(text):
    """
    Generate an embedding for the given text using the OpenAI model.
    """
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

# Generate embeddings if not already present
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

# Normalize question text for consistent matching.
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

# Debug: Run a simple query to verify that documents are in the index.
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

# SEARCH FUNCTIONS: SEMANTIC & VECTOR
def check_redis_cache(query):
    """Return cached answer if it exists."""
    try:
        cached_answer = redis_client.get(query)
        if cached_answer:
            print(f"✅ Using cached answer for query: {query}")
            return cached_answer
    except Exception as e:
        print(f"❌ Redis error: {e}")
    return None

def get_best_match(query):
    """
    Retrieve the best answer for the query by trying semantic then vector search.
    """
    cached_response = check_redis_cache(query)
    if cached_response:
        return cached_response

    # --- Semantic Search ---
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

    # --- Vector Search ---
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

async def get_response(user_query):
    """
    Retrieve a response by first trying search (semantic then vector),
    then falling back to GPT‑4o realtime if no match is found.
    """
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

# SPEECH RECOGNITION & SYNTHESIS SETUP
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "ar-EG"
speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"

def recognize_speech():
    """Listen for a single utterance using the default microphone."""
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
    """Convert the given text to speech and output via the default speaker."""
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Speech synthesis canceled:")
        print("  Reason: {}".format(cancellation.reason))
        print("  Error Details: {}".format(cancellation.error_details))

# HELPER: CLEAN TEXT FOR EXIT CHECK
def clean_text(text):
    """
    Remove common punctuation and whitespace from the beginning and end of the text,
    then convert to lower case.
    """
    return text.strip(" .،!؛؟").lower()

# CRITICAL ISSUE DETECTION
def detect_critical_issue(text):
    """
    Detect if the user's input contains a critical issue that should be passed to a human.
    """
    # Arabic Trigger Sentences
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

    # Get embedding for the input text
    text_embedding = np.array(get_embedding(text)).reshape(1, -1)

    # Calculate cosine similarity between the input text and trigger sentences
    similarities = cosine_similarity(text_embedding, trigger_embeddings)
    max_similarity = np.max(similarities)

    # If the similarity is above a threshold, consider it a critical issue
    if max_similarity > 0.9:
        print("This issue should be passed to a human.")
        return True
    return False

# ASYNCHRONOUS VOICE CHAT LOOP
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



@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    try:
        # Accept text input from the client
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_query = data.get("text")
        if not user_query:
            return jsonify({"error": "No text input provided"}), 400
        if clean_text(user_query) in ["إنهاء", "خروج"]:
            print("👋 Goodbye!")
            return jsonify({"response": "مع السلامة"})
        if detect_critical_issue(user_query):
            response = "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
            return jsonify({"response": response})

        # Process the query and generate a response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(get_response(user_query))
        loop.close()

        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /voice-chat: {e}")
        return jsonify({"error": "Internal server error"}), 500
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
