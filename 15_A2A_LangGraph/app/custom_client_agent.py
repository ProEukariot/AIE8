import asyncio
from uuid import uuid4
from typing import Optional

import httpx
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from app.tools import get_tool_belt
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

load_dotenv()


class CustomClientAgent:
    def __init__(self, general_agent_url: str = "http://localhost:10000"):
        self.name = "Custom Client Agent"
        self.agent = create_react_agent(
            model=ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo"),
            tools=get_tool_belt(),
        )
        self.general_agent_url = general_agent_url
        self._a2a_client: Optional[A2AClient] = None
        self._httpx_client: Optional[httpx.AsyncClient] = None

    async def _get_a2a_client(self) -> A2AClient:
        """Initialize or return existing A2A client for General Purpose Agent."""
        if self._a2a_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            resolver = A2ACardResolver(
                httpx_client=self._httpx_client,
                base_url=self.general_agent_url,
            )
            agent_card = await resolver.get_agent_card()
            self._a2a_client = A2AClient(
                httpx_client=self._httpx_client,
                agent_card=agent_card
            )
        return self._a2a_client

    async def _close_client(self):
        """Close HTTP client if it exists."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
            self._a2a_client = None

    def run(self, input: str) -> str:
        return self.agent.invoke({"messages": [("human", input)]})

    async def call_agent_async(self, message: str) -> str:
        """Call General Purpose Agent via HTTP API (async)."""
        client = await self._get_a2a_client()
        
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': message}],
                'message_id': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )
        
        response = await client.send_message(request)
        
        # Extract the text response from the A2A response
        # The response structure is: response.root.result.artifacts[0].parts[0].root.text
        try:
            result = response.root.result
            if result and hasattr(result, 'artifacts') and result.artifacts:
                # Get the first artifact's text content
                artifact = result.artifacts[0]
                if hasattr(artifact, 'parts') and artifact.parts:
                    part = artifact.parts[0]
                    if hasattr(part, 'root'):
                        root = part.root
                        # Check if root has a 'text' attribute (TextPart)
                        if hasattr(root, 'text') and root.text:
                            return root.text
                        # If root is a dict-like, try accessing 'text' key
                        if isinstance(root, dict) and 'text' in root:
                            return root['text']
            
            # Fallback: try to get messages from the response
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                if hasattr(result, 'messages') and result.messages:
                    # Extract text from messages
                    for msg in result.messages:
                        if hasattr(msg, 'parts') and msg.parts:
                            for msg_part in msg.parts:
                                if hasattr(msg_part, 'root') and hasattr(msg_part.root, 'text'):
                                    return msg_part.root.text
        except Exception as e:
            # If extraction fails, return the JSON dump
            pass
        
        # Final fallback: return string representation
        return str(response.model_dump(mode='json', exclude_none=True))

    def call_agent(self, other_agent_name: str, message: str) -> str:
        """Call General Purpose Agent via HTTP API (synchronous wrapper)."""
        return asyncio.run(self.call_agent_async(message))


if __name__ == "__main__":
    agent = CustomClientAgent()

    # print(agent.run("What's the weather in London?"))

    response = agent.call_agent("General Purpose Agent", "What's the weather in London?")
    print(response)
