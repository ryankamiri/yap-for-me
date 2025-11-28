from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
import logging
import uvicorn

from .config import BackendConfig
from .bluebubbles_client import BlueBubblesClient
from .context_manager import ContextManager
from .model_client import ModelClient
from .actions import ActionExecutor
from .webhooks import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="YapForMe Backend", version="1.0.0")

config = BackendConfig()
bluebubbles_client = BlueBubblesClient(config)
context_manager = ContextManager(config)
model_client = ModelClient(config)
action_executor = ActionExecutor(bluebubbles_client)

app.state.config = config
app.state.bluebubbles_client = bluebubbles_client
app.state.context_manager = context_manager
app.state.model_client = model_client
app.state.action_executor = action_executor

def get_context_manager_dep(request: Request) -> ContextManager:
    return request.app.state.context_manager


def get_action_executor_dep(request: Request) -> ActionExecutor:
    return request.app.state.action_executor


app.include_router(webhook_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/webhook/url")
async def get_webhook_url():
    """Get the webhook URL to configure in BlueBubbles"""
    return {
        "webhook_url": f"http://{config.backend_host}:{config.backend_port}/webhook/message",
        "event_subscriptions": ["New Messages"],
        "instructions": "Configure this URL in BlueBubbles Server > API & Webhooks > Add Webhook"
    }


@app.get("/chats/{chat_guid}/context")
async def get_chat_context(
    chat_guid: str,
    context_manager: ContextManager = Depends(get_context_manager_dep)
):
    context = context_manager.get_context(chat_guid)
    messages = context_manager.get_messages(chat_guid)
    
    return {
        "chat_guid": chat_guid,
        "context": context,
        "message_count": len(messages),
        "messages": [
            {
                "timestamp": msg.timestamp,
                "speaker": msg.speaker,
                "text": msg.text,
                "message_guid": msg.message_guid
            }
            for msg in messages
        ]
    }


@app.post("/chats/{chat_guid}/actions")
async def trigger_action(
    chat_guid: str,
    action_data: dict,
    action_executor: ActionExecutor = Depends(get_action_executor_dep)
):
    action_type = action_data.get("action_type")
    params = action_data.get("params", {})
    params["chat_guid"] = chat_guid
    
    result = await action_executor.execute_action(action_type, params)
    
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.backend_host,
        port=config.backend_port,
        reload=True
    )

