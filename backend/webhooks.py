from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
import logging

from .context_manager import ContextManager
from .model_client import ModelClient
from .bluebubbles_client import BlueBubblesClient

logger = logging.getLogger(__name__)

router = APIRouter()


def get_context_manager(request: Request) -> ContextManager:
    return request.app.state.context_manager


def get_model_client(request: Request) -> ModelClient:
    return request.app.state.model_client


def get_bluebubbles_client(request: Request) -> BlueBubblesClient:
    return request.app.state.bluebubbles_client


class MessageEvent(BaseModel):
    chat_guid: str
    message_guid: str
    sender: str
    sender_name: Optional[str] = None
    text: str
    timestamp: str
    type: str
    replying_to: Optional[str] = None


@router.post("/webhook/message")
async def handle_message_webhook(
    request: Request,
    context_manager: ContextManager = Depends(get_context_manager),
    model_client: ModelClient = Depends(get_model_client),
    bluebubbles_client: BlueBubblesClient = Depends(get_bluebubbles_client)
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
        
        context = context_manager.get_context(chat_guid)
        
        if not context:
            logger.info(f"Chat {chat_guid} not cached, fetching recent messages from BlueBubbles")
            try:
                offset = 0
                limit = 100
                needs_more = True
                
                while needs_more:
                    messages = await bluebubbles_client.get_chat_messages(chat_guid, limit=limit, offset=offset)
                    if not messages:
                        break
                    
                    needs_more = context_manager.populate_from_bluebubbles_messages(chat_guid, messages)
                    offset += len(messages)
                    
                    if len(messages) < limit:
                        break
                
                context = context_manager.get_context(chat_guid)
                logger.info(f"Populated context for chat {chat_guid} with messages up to offset {offset}")
            except Exception as e:
                logger.warning(f"Failed to fetch messages from BlueBubbles for chat {chat_guid}: {str(e)}")
        
        message_data_dict = {
            "timestamp": timestamp,
            "speaker": sender_name,
            "text": text,
            "message_guid": message_guid,
            "replying_to": replying_to
        }
        
        context_manager.add_message(chat_guid, message_data_dict)
        
        messages = context_manager.get_messages(chat_guid)
        current_message = messages[-1]
        new_message_text = context_manager.format_message(current_message)
        
        response_prefix = context_manager.format_response_prefix(chat_guid, replying_to=replying_to)
        
        model_output = await model_client.infer(context, new_message_text, response_prefix)
        
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
        
        return {
            "status": "processed",
            "chat_guid": chat_guid,
            "message_guid": message_guid,
            "model_output": model_output
        }
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

