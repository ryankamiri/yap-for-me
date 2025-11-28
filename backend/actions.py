from typing import Dict, Any
from .bluebubbles_client import BlueBubblesClient
import logging

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(self, bluebubbles_client: BlueBubblesClient):
        self.client = bluebubbles_client
    
    async def execute_action(
        self,
        action_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if action_type == "react":
                return await self._react(params)
            elif action_type == "reply":
                return await self._reply(params)
            elif action_type == "send_message":
                return await self._send_message(params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _react(self, params: Dict[str, Any]) -> Dict[str, Any]:
        chat_guid = params.get("chat_guid")
        message_guid = params.get("message_guid")
        reaction_type = params.get("reaction_type")
        
        if not chat_guid or not message_guid or not reaction_type:
            return {
                "success": False,
                "error": "Missing required parameters: chat_guid, message_guid, reaction_type"
            }
        
        result = await self.client.react_to_message(chat_guid, message_guid, reaction_type)
        return {"success": True, "result": result}
    
    async def _reply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        chat_guid = params.get("chat_guid")
        message_guid = params.get("message_guid")
        text = params.get("text")
        
        if not chat_guid or not message_guid or not text:
            return {
                "success": False,
                "error": "Missing required parameters: chat_guid, message_guid, text"
            }
        
        result = await self.client.reply_to_message(chat_guid, message_guid, text)
        return {"success": True, "result": result}
    
    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        chat_guid = params.get("chat_guid")
        text = params.get("text")
        
        if not chat_guid or not text:
            return {
                "success": False,
                "error": "Missing required parameters: chat_guid, text"
            }
        
        result = await self.client.send_message(chat_guid, text)
        return {"success": True, "result": result}

