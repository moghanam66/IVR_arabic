import os
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount
from aiohttp import web
from botbuilder.integration.aiohttp import BotFrameworkAdapter, BotFrameworkAdapterSettings
import asyncio
#dd
# Load environment variables
APP_ID = os.getenv("b0a29017-ea3f-4697-aef7-0cb05979d16c", "")
APP_PASSWORD = os.getenv("2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ", "")

# Bot Class
class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        await turn_context.send_activity(MessageFactory.text(f"You said: {turn_context.activity.text}"))
    
    async def on_members_added_activity(self, members_added, turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")

# Adapter Settings
settings = BotFrameworkAdapterSettings(app_id=APP_ID, app_password=APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

bot = MyBot()

async def messages(req):
    body = await req.json()
    activity = TurnContext.deserialize(body)
    auth_header = req.headers.get("Authorization", "")
    response = await adapter.process_activity(activity, auth_header, bot.on_turn)
    return web.json_response(response)

app = web.Application()
app.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=3978)
