from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import logging
import re

from .context_manager import ContextManager
from .model_client import ModelClient
from .bluebubbles_client import BlueBubblesClient
from .debounce_manager import DebounceManager
from .actions import ActionExecutor

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


def get_action_executor(request: Request) -> ActionExecutor:
    return request.app.state.action_executor


def parse_tool_calls(model_output: str) -> List[Dict[str, Any]]:
    """Parse code-style tool calls from model output.
    
    Extracts tool calls in format:
    - react(message_guid="...", reaction_type="...")
    - reply(message_guid="...", text="...")
    - send_message(text="...")
    
    Returns:
        List of dicts with 'action_type' and 'params' keys
    """
    tool_calls = []
    
    lines = model_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        react_match = re.match(r'react\s*\(\s*message_guid\s*=\s*"([^"]+)"\s*,\s*reaction_type\s*=\s*"([^"]+)"\s*\)', line)
        if react_match:
            tool_calls.append({
                'action_type': 'react',
                'params': {
                    'message_guid': react_match.group(1),
                    'reaction_type': react_match.group(2)
                }
            })
            continue
        
        reply_match = re.match(r'reply\s*\(\s*message_guid\s*=\s*"([^"]+)"\s*,\s*text\s*=\s*(.+?)\s*\)', line, re.DOTALL)
        if reply_match:
            text_value = reply_match.group(2).strip()
            if text_value.startswith('"') and text_value.endswith('"'):
                text_value = text_value[1:-1]
            elif text_value.startswith("'") and text_value.endswith("'"):
                text_value = text_value[1:-1]
            tool_calls.append({
                'action_type': 'reply',
                'params': {
                    'message_guid': reply_match.group(1),
                    'text': text_value
                }
            })
            continue
        
        send_msg_match = re.match(r'send_message\s*\(\s*text\s*=\s*(.+?)\s*\)', line, re.DOTALL)
        if send_msg_match:
            text_value = send_msg_match.group(1).strip()
            if text_value.startswith('"') and text_value.endswith('"'):
                text_value = text_value[1:-1]
            elif text_value.startswith("'") and text_value.endswith("'"):
                text_value = text_value[1:-1]
            tool_calls.append({
                'action_type': 'send_message',
                'params': {
                    'text': text_value
                }
            })
            continue
    
    return tool_calls


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
    context_manager: ContextManager,
    model_client: ModelClient,
    bluebubbles_client: BlueBubblesClient,
    action_executor: ActionExecutor
) -> None:
    """Process a message after debounce period."""
    try:
        context = await context_manager.get_context_or_fetch(chat_guid, bluebubbles_client)
        
        if not context:
            logger.warning(f"No context available for chat {chat_guid}, skipping processing")
            return
        
        full_prompt = context
        
        model_output = await model_client.infer(full_prompt)
        
        logger.info(f"Model output for chat {chat_guid}: {model_output}")
        
        tool_calls = parse_tool_calls(model_output)
        
        if not tool_calls:
            logger.warning(f"No tool calls parsed from model output for chat {chat_guid}")
            return
        
        for tool_call in tool_calls:
            action_type = tool_call.get('action_type')
            params = tool_call.get('params', {})
            params['chat_guid'] = chat_guid
            
            logger.info(f"Executing action {action_type} for chat {chat_guid} with params: {params}")
            
            result = await action_executor.execute_action(action_type, params)
            
            if result.get('success'):
                logger.info(f"Successfully executed {action_type} for chat {chat_guid}")
            else:
                logger.error(f"Failed to execute {action_type} for chat {chat_guid}: {result.get('error')}")
    
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
        
        action_executor = get_action_executor(request)
        
        await debounce_manager.debounce(
            chat_guid,
            _process_message,
            chat_guid=chat_guid,
            context_manager=context_manager,
            model_client=model_client,
            bluebubbles_client=bluebubbles_client,
            action_executor=action_executor
        )
        
        return {
            "status": "queued",
            "chat_guid": chat_guid,
            "message_guid": message_guid
        }
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

