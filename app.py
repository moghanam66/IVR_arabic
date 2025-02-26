import os
from flask import Flask, request, jsonify
from botbuilder.core import BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity

# Initialize Flask app
app = Flask(__name__)

# Bot Framework Adapter (used to process messages)
adapter = BotFrameworkAdapter(
    app_id=os.getenv("b0a29017-ea3f-4697-aef7-0cb05979d16c"), 
    app_password=os.getenv("2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ")
)

# Process incoming messages
@app.route("/api/messages", methods=["POST"])
async def messages():
    body = await request.get_json()
    activity = Activity().deserialize(body)
    
    async def on_turn(turn_context: TurnContext):
        if activity.type == "message":
            # Echo back the user's message
            await turn_context.send_activity(f"You said: {turn_context.activity.text}")

    await adapter.process_activity(activity, request.headers.get("Authorization"), on_turn)
    return jsonify({})

if __name__ == "__main__":
    app.run(port=3978)
