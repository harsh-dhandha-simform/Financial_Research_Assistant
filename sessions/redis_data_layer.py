import json
import logging
from typing import Optional, Dict, List
import uuid
from datetime import datetime
import hashlib
import os

import redis.asyncio as redis
from chainlit.data.base import BaseDataLayer
from chainlit.types import Feedback, PaginatedResponse, Pagination, ThreadDict, ThreadFilter
from chainlit.user import PersistedUser, User

logger = logging.getLogger(__name__)

class RedisDataLayer(BaseDataLayer):
    """
    A custom Chainlit DataLayer backed by Redis.
    Provides persistence for users, threads (chat history), steps, and elements
    to power the Chainlit sidebar history natively.
    """
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Initialized RedisDataLayer at {redis_url}")

    @staticmethod
    def hash_password(password: str) -> str:
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + ':' + key.hex()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        try:
            salt_hex, key_hex = hashed.split(':')
            salt = bytes.fromhex(salt_hex)
            key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return key.hex() == key_hex
        except Exception:
            return False
        
    # --- Users ---
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        data = await self.client.get(f"cl_user:{identifier}")
        if data:
            u_dict = json.loads(data)
            return PersistedUser(
                id=u_dict["id"], 
                identifier=u_dict["identifier"], 
                metadata=u_dict.get("metadata", {}), 
                createdAt=u_dict.get("createdAt", "")
            )
        return None

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        uid = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        p_user = PersistedUser(id=uid, identifier=user.identifier, metadata=user.metadata, createdAt=created_at)
        await self.client.set(f"cl_user:{user.identifier}", json.dumps({
            "id": uid, "identifier": user.identifier, "metadata": user.metadata, "createdAt": created_at
        }))
        return p_user
        
    # --- Threads (Chat History) ---
    async def get_thread_author(self, thread_id: str) -> str:
        data = await self.client.get(f"cl_thread:{thread_id}")
        if data:
            return json.loads(data).get("userId", "")
        return ""

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        data = await self.client.get(f"cl_thread:{thread_id}")
        if data:
            return json.loads(data)
        return None

    async def update_thread(
        self, 
        thread_id: str, 
        name: Optional[str] = None, 
        user_id: Optional[str] = None, 
        metadata: Optional[Dict] = None, 
        tags: Optional[List[str]] = None
    ):
        data = await self.client.get(f"cl_thread:{thread_id}")
        thread_dict = json.loads(data) if data else {"id": thread_id, "createdAt": datetime.utcnow().isoformat(), "steps": [], "elements": []}
        
        if name is not None:
            thread_dict["name"] = name
        if user_id is not None:
            thread_dict["userIdentifier"] = user_id
            thread_dict["userId"] = user_id
            # Add to user's sorted set of threads (score = timestamp)
            await self.client.zadd(f"cl_user_threads:{user_id}", {thread_id: datetime.utcnow().timestamp()})
            
        if metadata is not None:
            thread_dict["metadata"] = metadata
        if tags is not None:
            thread_dict["tags"] = tags
            
        await self.client.set(f"cl_thread:{thread_id}", json.dumps(thread_dict))

    async def delete_thread(self, thread_id: str):
        thread = await self.get_thread(thread_id)
        if thread and thread.get("userIdentifier"):
            await self.client.zrem(f"cl_user_threads:{thread.get('userIdentifier')}", thread_id)
        await self.client.delete(f"cl_thread:{thread_id}")
        
    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """Fetch paginated chat history for the left sidebar."""
        f_dict = filters.model_dump() if hasattr(filters, "model_dump") else filters.dict() if hasattr(filters, "dict") else {}
        user_id = f_dict.get("userIdentifier") or f_dict.get("userId")
            
        if not user_id:
            return PaginatedResponse(data=[], pageInfo={"hasNextPage": False, "startCursor": None, "endCursor": None})
            
        # Get thread IDs for this user, sorted by newest first
        thread_ids = await self.client.zrevrange(f"cl_user_threads:{user_id}", 0, -1)
        
        threads = []
        for tid in thread_ids:
            t = await self.get_thread(tid)
            if t:
                # For the listing, Chainlit expects threads without the full steps payload
                listing_thread = dict(t)
                listing_thread.pop("steps", None)
                listing_thread.pop("elements", None)
                threads.append(listing_thread)
                
        # Basic pagination cursor
        start = 0
        if pagination.cursor:
            try:
                start = next(i for i, t in enumerate(threads) if t["id"] == pagination.cursor) + 1
            except StopIteration:
                pass
                
        end = start + (pagination.first or 20)
        page_data = threads[start:end]
        has_next = end < len(threads)
        
        return PaginatedResponse(
            data=page_data,
            pageInfo={
                "hasNextPage": has_next, 
                "startCursor": page_data[0]["id"] if page_data else None,
                "endCursor": page_data[-1]["id"] if page_data else None
            }
        )

    # --- Steps (Messages) ---
    async def create_step(self, step_dict: Dict):
        thread_id = step_dict.get("threadId")
        if thread_id:
            thread = await self.get_thread(thread_id)
            if thread:
                if "steps" not in thread:
                    thread["steps"] = []
                thread["steps"].append(step_dict)
                await self.client.set(f"cl_thread:{thread_id}", json.dumps(thread))

    async def update_step(self, step_dict: Dict):
        thread_id = step_dict.get("threadId")
        if thread_id:
            thread = await self.get_thread(thread_id)
            if thread and "steps" in thread:
                for i, s in enumerate(thread["steps"]):
                    if s.get("id") == step_dict.get("id"):
                        thread["steps"][i] = step_dict
                        break
                await self.client.set(f"cl_thread:{thread_id}", json.dumps(thread))

    async def delete_step(self, step_id: str):
        pass # Optional/Simplified

    # --- Elements (Attachments/Sources) ---
    async def create_element(self, element):
        element_dict = element.to_dict()
        thread_id = element_dict.get("threadId")
        if thread_id:
            thread = await self.get_thread(thread_id)
            if thread:
                if "elements" not in thread:
                    thread["elements"] = []
                thread["elements"].append(element_dict)
                await self.client.set(f"cl_thread:{thread_id}", json.dumps(thread))

    async def get_element(self, thread_id: str, element_id: str) -> Optional[Dict]:
        thread = await self.get_thread(thread_id)
        if thread and "elements" in thread:
            for el in thread["elements"]:
                if el.get("id") == element_id:
                    return el
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass # Optional/Simplified

    # --- Feedback (Thumbs up/down) ---
    async def upsert_feedback(self, feedback: Feedback) -> str:
        fid = feedback.id or str(uuid.uuid4())
        await self.client.set(f"cl_feedback:{fid}", json.dumps({"value": feedback.value, "comment": feedback.comment}))
        return fid

    async def delete_feedback(self, feedback_id: str) -> bool:
        await self.client.delete(f"cl_feedback:{feedback_id}")
        return True
        
    # --- Other Requirements ---
    async def get_favorite_steps(self, user_id: str) -> List[Dict]:
        return []

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        await self.client.aclose()
