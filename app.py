from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from flask import Flask, request, jsonify
from botbuilder.schema import Activity
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
# Define the app and adapter settings
app = Flask(__name__)
 
# Set up BotFrameworkAdapter with app credentials (replace with your actual app ID and password)
APP_ID = "b0a29017-ea3f-4697-aef7-0cb05979d16c"
APP_PASSWORD = "2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ"
adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)
 
# Define a bot class that uses your get_response logic
class MyBot:
    async def on_turn(self, turn_context: TurnContext):

             turn_context.send_activity(f"Received activity of type: {turn_context.activity.type}")
 
# Create an instance of the bot

# Create the bot instance
bot = MyBot()
 
# Error handling for adapter
async def on_error(context, error):
    print(f"Error: {error}")
    await context.send_activity("Oops! Something went wrong.")
    return True
 
adapter.on_turn_error = on_error
 
# Define the route to handle incoming messages
@app.route("/api/messages", methods=["POST"])
async def messages():
    # Parse the incoming request as an activity
    if "application/json" not in request.headers["Content-Type"]:
        return jsonify({"error": "Invalid Content-Type, must be application/json"}), 415
 
    body = await request.json
    activity = Activity().deserialize(body)
 
    # Route the activity to the BotFrameworkAdapter for processing
    response = await adapter.process_activity(activity, "", bot.on_turn)
    return jsonify({"status": "ok"})
 
if __name__ == "__main__":
    app.run(debug=True)
