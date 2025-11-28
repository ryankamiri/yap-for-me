from typing import Dict, Callable, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class DebounceManager:
    """Manages debouncing of message processing per chat.
    
    When multiple messages arrive quickly, only processes the most recent one
    after a period of inactivity.
    """
    
    def __init__(self, debounce_seconds: float = 2.5):
        """Initialize debounce manager.
        
        Args:
            debounce_seconds: Number of seconds to wait after last message before processing
        """
        self.debounce_seconds = debounce_seconds
        self.pending_tasks: Dict[str, asyncio.Task] = {}
        self.pending_data: Dict[str, Dict[str, Any]] = {}
    
    async def debounce(
        self,
        chat_guid: str,
        process_func: Callable,
        *args,
        **kwargs
    ) -> None:
        """Schedule processing with debouncing.
        
        If a task is already pending for this chat, it will be cancelled
        and a new one will be scheduled.
        
        Args:
            chat_guid: Chat identifier
            process_func: Async function to call after debounce period
            *args, **kwargs: Arguments to pass to process_func
        """
        if chat_guid in self.pending_tasks:
            task = self.pending_tasks[chat_guid]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.pending_data[chat_guid] = {
            "func": process_func,
            "args": args,
            "kwargs": kwargs
        }
        
        task = asyncio.create_task(
            self._debounced_process(chat_guid)
        )
        self.pending_tasks[chat_guid] = task
    
    async def _debounced_process(self, chat_guid: str) -> None:
        """Internal method that waits and then processes."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            
            if chat_guid not in self.pending_data:
                return
            
            data = self.pending_data.pop(chat_guid)
            func = data["func"]
            args = data["args"]
            kwargs = data["kwargs"]
            
            await func(*args, **kwargs)
            
        except asyncio.CancelledError:
            logger.debug(f"Debounce task cancelled for chat {chat_guid}")
        except Exception as e:
            logger.error(f"Error in debounced processing for chat {chat_guid}: {str(e)}", exc_info=True)
        finally:
            if chat_guid in self.pending_tasks:
                del self.pending_tasks[chat_guid]
            if chat_guid in self.pending_data:
                del self.pending_data[chat_guid]
    
    def cancel_all(self) -> None:
        """Cancel all pending tasks (useful for shutdown)."""
        for task in self.pending_tasks.values():
            if not task.done():
                task.cancel()

