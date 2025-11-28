from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
import logging

from .context_manager import ContextManager
from .model_client import ModelClient
from .bluebubbles_client import BlueBubblesClient
from .debounce_manager import DebounceManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_context_manager(request: Request) -> ContextManager:
    return request.app.state.context_manager


def get_model_client(request: Request) -> ModelClient:
    return request.app.state.model_client


def get_bluebubbles_client(request: Request) -> BlueBubblesClient:
    return request.app.state.bluebubbles_client


def get_debounce_manager(request: Request) -> DebounceManager:
    return request.app.state.debounce_manager


class MessageEvent(BaseModel):
    chat_guid: str
    message_guid: str
    sender: str
    sender_name: Optional[str] = None
    text: str
    timestamp: str
    type: str
    replying_to: Optional[str] = None


async def _process_message(
    chat_guid: str,
    replying_to: Optional[str],
    context_manager: ContextManager,
    model_client: ModelClient,
    bluebubbles_client: BlueBubblesClient
) -> None:
    """Process a message after debounce period."""
    try:
        context = await context_manager.get_context_or_fetch(chat_guid, bluebubbles_client)
        
        if not context:
            logger.warning(f"No context available for chat {chat_guid}, skipping processing")
            return
        
        response_prefix = context_manager.format_response_prefix(chat_guid, replying_to=replying_to)
        
        full_prompt = f"{context}\n{response_prefix}"
        
        model_output = await model_client.infer(full_prompt)
        
        logger.info(f"Model output for chat {chat_guid}: {model_output}")
        
        # TODO: TEMPORARY SOLUTION - Replace with proper tool calling extraction
        # Currently, we just send the raw model output as text. This needs to be replaced
        # with efficient tool calling extraction that parses the model output to extract
        # structured actions (send, reply, react, etc.) and executes them appropriately.
        if model_output.strip():
            try:
                await bluebubbles_client.send_message(chat_guid, model_output.strip())
                logger.info(f"Sent model output to chat {chat_guid}")
            except Exception as e:
                logger.error(f"Failed to send message to chat {chat_guid}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in debounced message processing: {str(e)}", exc_info=True)


@router.post("/webhook/message")
async def handle_message_webhook(
    request: Request,
    context_manager: ContextManager = Depends(get_context_manager),
    model_client: ModelClient = Depends(get_model_client),
    bluebubbles_client: BlueBubblesClient = Depends(get_bluebubbles_client),
    debounce_manager: DebounceManager = Depends(get_debounce_manager)
):
    try:
        data = await request.json()
        logger.debug(f"Received webhook data: {data}")
        
        event_type = data.get("event", "").lower()
        
        supported_events = ["new-message", "message-received", "new-messages"]
        if event_type not in supported_events:
            logger.info(f"Ignoring event type: {event_type}")
            return {"status": "ignored", "reason": f"Event type {event_type} not handled"}
        
        message_data = data.get("data", {})
        if not message_data:
            logger.warning("Webhook data missing 'data' field")
            return {"status": "error", "reason": "Missing data field"}
        
        message_type = message_data.get("type", "")
        if message_type != "Incoming":
            logger.info(f"Ignoring message type: {message_type}")
            return {"status": "ignored", "reason": "Only processing incoming messages"}
        
        chat_guid = message_data.get("chatGuid", "")
        message_guid = message_data.get("guid", "")
        text = message_data.get("text", "")
        timestamp = message_data.get("dateCreated", "")
        
        handle = message_data.get("handle", {})
        if handle:
            sender_id = handle.get("id", "Unknown")
            sender_name = handle.get("name", sender_id)
        else:
            sender_id = "Unknown"
            sender_name = "Unknown"
        
        replying_to = message_data.get("replyToGuid")
        
        if not chat_guid or not message_guid:
            logger.warning(f"Missing required fields: chat_guid={chat_guid}, message_guid={message_guid}")
            return {"status": "error", "reason": "Missing chat_guid or message_guid"}
        
        message_data_dict = {
            "timestamp": timestamp,
            "speaker": sender_name,
            "text": text,
            "message_guid": message_guid,
            "replying_to": replying_to
        }
        
        context_manager.add_message(chat_guid, message_data_dict)
        
        await debounce_manager.debounce(
            chat_guid,
            _process_message,
            chat_guid=chat_guid,
            replying_to=replying_to,
            context_manager=context_manager,
            model_client=model_client,
            bluebubbles_client=bluebubbles_client
        )
        
        return {
            "status": "queued",
            "chat_guid": chat_guid,
            "message_guid": message_guid
        }
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

