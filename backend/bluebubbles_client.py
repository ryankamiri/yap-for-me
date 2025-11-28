import httpx
from typing import Dict, List, Optional, Any, Literal, get_args
from .config import BackendConfig


ReactionType = Literal[
    "love", "like", "dislike", "laugh", "emphasize", "question",
    "-love", "-like", "-dislike", "-laugh", "-emphasize", "-question"
]

VALID_REACTIONS = set(get_args(ReactionType))


class BlueBubblesError(Exception):
    """Exception raised for BlueBubbles API errors"""
    def __init__(self, message: str, status: int = None, data: Dict = None):
        self.message = message
        self.status = status
        self.data = data
        super().__init__(self.message)


class BlueBubblesClient:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.base_url = config.bluebubbles_url.rstrip("/")
        self.password = config.bluebubbles_password
        
        self.headers = {"Content-Type": "application/json"}
        self.timeout = httpx.Timeout(30.0)
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        params["password"] = self.password
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                json=json_data,
                params=params
            )
            status_code = response.status_code
            response.raise_for_status()
            result = response.json()
            
            return status_code, result
    
    async def react_to_message(
        self,
        chat_guid: str,
        message_guid: str,
        reaction_type: ReactionType
    ) -> Dict[str, Any]:
        if reaction_type not in VALID_REACTIONS:
            raise ValueError(
                f"Invalid reaction_type: {reaction_type}. "
                f"Must be one of: {', '.join(sorted(VALID_REACTIONS))}"
            )
        
        _, response = await self._request(
            "POST",
            f"/api/v1/message/{message_guid}/reaction",
            json_data={
                "action": "react",
                "chatGuid": chat_guid,
                "reaction": reaction_type
            }
        )
        return response.get("data", {})
    
    async def reply_to_message(
        self,
        chat_guid: str,
        message_guid: str,
        text: str
    ) -> Dict[str, Any]:
        _, response = await self._request(
            "POST",
            "/api/v1/message",
            json_data={
                "chatGuid": chat_guid,
                "text": text,
                "replyToGuid": message_guid
            }
        )
        return response.get("data", {})
    
    async def send_message(self, chat_guid: str, text: str) -> Dict[str, Any]:
        _, response = await self._request(
            "POST",
            "/api/v1/message",
            json_data={
                "chatGuid": chat_guid,
                "text": text
            }
        )
        return response.get("data", {})
    
    async def get_message(self, chat_guid: str, message_guid: str) -> Dict[str, Any]:
        _, response = await self._request(
            "GET",
            f"/api/v1/message/{message_guid}",
            params={"chatGuid": chat_guid}
        )
        return response.get("data", {})
    
    async def get_chat_messages(
        self,
        chat_guid: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        _, response = await self._request(
            "GET",
            f"/api/v1/chat/{chat_guid}/messages",
            params={"limit": limit, "offset": offset}
        )
        return response.get("data", [])

