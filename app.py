from flask import Flask, request, jsonify, Response
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity

app = Flask(__name__)

# Bot Framework credentials (replace with your actual values)
MICROSOFT_APP_ID = "b0a29017-ea3f-4697-aef7-0cb05979d16c"
MICROSOFT_APP_PASSWORD = "2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ"

# Initialize Bot Framework adapter
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Define a simple bot class
class SimpleBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == "message":
            await turn_context.send_activity("welcome")

bot = SimpleBot()

@app.route("/api/messages", methods=["POST"])
def messages():
    if request.headers.get("Content-Type", "") != "application/json":
        return Response("Invalid Content-Type", status=415)

    try:
        body = request.json
        activity = Activity().deserialize(body)

        async def process_activity():
            await adapter.process_activity(activity, "", bot.on_turn)

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_activity())
        loop.close()
        
        return Response(status=201)

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

if __name__ == "__main__":
    app.run()
