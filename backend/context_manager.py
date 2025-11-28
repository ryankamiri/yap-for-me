from typing import Dict, List, Optional
from datetime import datetime
from .config import BackendConfig


class Message:
    def __init__(
        self,
        timestamp: str,
        speaker: str,
        text: str,
        message_guid: Optional[str] = None,
        replying_to: Optional[str] = None
    ):
        self.timestamp = timestamp
        self.speaker = speaker
        self.text = text
        self.message_guid = message_guid
        self.replying_to = replying_to


class ContextManager:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.contexts: Dict[str, List[Message]] = {}
        model_config = config.get_model_config()
        inference_config = config.get_inference_config()
        
        if "max_length" not in model_config:
            raise ValueError("max_length is required in model config")
        self.max_tokens = model_config["max_length"]
        
        if "outgoing_speaker_name" not in inference_config:
            raise ValueError("outgoing_speaker_name is required in inference config")
        self.outgoing_speaker_name = inference_config["outgoing_speaker_name"]
        
        self.context_window_tokens = int(self.max_tokens * 1.1)
    
    def add_message(
        self,
        chat_guid: str,
        message_data: Dict[str, any]
    ) -> None:
        if chat_guid not in self.contexts:
            self.contexts[chat_guid] = []
        
        timestamp = message_data.get("timestamp", "")
        speaker = message_data.get("speaker", "")
        text = message_data.get("text", "")
        message_guid = message_data.get("message_guid")
        replying_to = message_data.get("replying_to")
        
        message = Message(
            timestamp=timestamp,
            speaker=speaker,
            text=text,
            message_guid=message_guid,
            replying_to=replying_to
        )
        
        self.contexts[chat_guid].append(message)
        self._trim_context(chat_guid)
    
    def format_message(self, message: Message) -> str:
        guid_part = f"[guid:{message.message_guid}]" if message.message_guid else ""
        if message.replying_to:
            return f"{guid_part}[{message.timestamp}] {message.speaker}: {message.replying_to} {message.text}"
        return f"{guid_part}[{message.timestamp}] {message.speaker}: {message.text}"
    
    def format_context(self, messages: List[Message]) -> str:
        return "\n".join([self.format_message(msg) for msg in messages])
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _trim_context(self, chat_guid: str) -> None:
        if chat_guid not in self.contexts:
            return
        
        messages = self.contexts[chat_guid]
        if not messages:
            return
        
        total_tokens = self._estimate_tokens(self.format_context(messages))
        
        while total_tokens > self.context_window_tokens and len(messages) > 1:
            messages.pop(0)
            total_tokens = self._estimate_tokens(self.format_context(messages))
    
    def get_context(self, chat_guid: str) -> str:
        if chat_guid not in self.contexts:
            return ""
        
        messages = self.contexts[chat_guid]
        
        if not messages:
            return ""
        
        return self.format_context(messages)
    
    def clear_context(self, chat_guid: str) -> None:
        if chat_guid in self.contexts:
            del self.contexts[chat_guid]
    
    def get_messages(self, chat_guid: str) -> List[Message]:
        return self.contexts.get(chat_guid, [])
    
    def populate_from_bluebubbles_messages(
        self,
        chat_guid: str,
        bluebubbles_messages: List[Dict]
    ) -> bool:
        if chat_guid not in self.contexts:
            self.contexts[chat_guid] = []
        
        for msg_data in bluebubbles_messages:
            text = msg_data.get("text", "")
            if not text:
                continue
            
            message_guid = msg_data.get("guid", "")
            date_created = msg_data.get("dateCreated", "")
            is_from_me = msg_data.get("isFromMe", False)
            reply_to_guid = msg_data.get("replyToGuid")
            
            handle = msg_data.get("handle", {})
            if is_from_me:
                speaker = self.outgoing_speaker_name
            elif handle:
                speaker = handle.get("name") or handle.get("address") or "Unknown"
            else:
                speaker = "Unknown"
            
            timestamp = self._format_timestamp(date_created)
            
            message = Message(
                timestamp=timestamp,
                speaker=speaker,
                text=text,
                message_guid=message_guid,
                replying_to=reply_to_guid
            )
            
            self.contexts[chat_guid].insert(0, message)
            
            total_tokens = self._estimate_tokens(self.format_context(self.contexts[chat_guid]))
            if total_tokens >= self.context_window_tokens:
                return False
        
        return True
    
    def _format_timestamp(self, date_created: any) -> str:
        if not isinstance(date_created, (int, float)):
            raise ValueError(f"dateCreated must be a number (milliseconds), got {type(date_created)}")
        dt = datetime.fromtimestamp(date_created / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

