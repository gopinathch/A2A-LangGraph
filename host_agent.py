import logging
import uuid
from typing import List, Optional
import os

from dotenv import load_dotenv

import requests
import typer
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI  # Changed from ChatOpenAI to AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


load_dotenv()

logger = logging.getLogger(__name__)

class AgentCapabilities:
    def __init__(
        self, streaming=False, pushNotifications=False, stateTransitionHistory=False
    ):
        self.streaming = streaming
        self.pushNotifications = pushNotifications
        self.stateTransitionHistory = stateTransitionHistory


class AgentCard:
    def __init__(
        self,
        name: str,
        url: str,
        version: str,
        capabilities: AgentCapabilities,
        description: Optional[str] = None,
    ):
        self.name = name
        self.url = url
        self.version = version
        self.capabilities = capabilities
        self.description = description or "No description."


class TaskState:
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"
    INPUT_REQUIRED = "input-required"


###############################################################################
# 2) Synchronous RemoteAgentClient
###############################################################################
class RemoteAgentClient:
    """Communicates with a single remote agent (A2A) in synchronous mode."""

    def __init__(self, base_url: str):
        logger.info(f"Initializing RemoteAgentClient with base_url: {base_url}")
        self.base_url = base_url
        self.agent_card: Optional[AgentCard] = None

    def fetch_agent_card(self) -> AgentCard:
        """GET /.well-known/agent.json to retrieve the remote agent's card."""
        logger.info(f"Fetching agent card from {self.base_url}")
        url = f"{self.base_url}/.well-known/agent.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        caps_data = data["capabilities"]
        caps = AgentCapabilities(**caps_data)

        card = AgentCard(
            name=data["name"],
            url=self.base_url,
            version=data["version"],
            capabilities=caps,
            description=data.get("description", ""),
        )
        self.agent_card = card
        logger.info(f"Successfully fetched agent card: {card.name}, version: {card.version}")
        return card

    def send_task(self, task_id: str, session_id: str, message_text: str) -> dict:
        """POST / with JSON-RPC request: method=tasks/send."""
        logger.info(f"Sending task {task_id} to {self.base_url} with session {session_id}")
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message_text}],
                },
            },
        }
        logger.info(f"Sending payload to {self.base_url}")
        r = requests.post(self.base_url, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp and resp["error"] is not None:
            logger.error(f"Remote agent error: {resp['error']}")
            raise RuntimeError(f"Remote agent error: {resp['error']}")
        logger.info(f"Received successful response for task {task_id}")
        return resp.get("result", {})


class HostAgent:
    """Holds references to multiple RemoteAgentClients, one per address."""

    def __init__(self, remote_addresses: List[str]):
        logger.info(f"Initializing HostAgent with {len(remote_addresses)} remote addresses")
        self.clients = {}
        for addr in remote_addresses:
            logger.info(f"Creating client for address: {addr}")
            self.clients[addr] = RemoteAgentClient(addr)

    def initialize(self):
        """Fetch agent cards for all addresses (synchronously)."""
        logger.info("Initializing all remote agent clients")
        for addr, client in self.clients.items():
            logger.info(f"Initializing client for address: {addr}")
            client.fetch_agent_card()
        logger.info("All remote agent clients initialized successfully")

    def list_agents_info(self) -> list:
        """Return a list of {name, description, url, streaming} for each loaded agent."""
        logger.info("Listing all available agents")
        infos = []
        for addr, c in self.clients.items():
            card = c.agent_card
            if card:
                logger.info(f"Found agent: {card.name} at {card.url}")
                infos.append(
                    {
                        "name": card.name,
                        "description": card.description,
                        "url": card.url,
                        "streaming": card.capabilities.streaming,
                    }
                )
            else:
                logger.warning(f"No agent card found for address: {addr}")
                infos.append(
                    {
                        "name": "Unknown",
                        "description": "Not loaded",
                        "url": addr,
                        "streaming": False,
                    }
                )
        logger.info(f"Found {len(infos)} agents")
        return infos

    def get_client_by_name(self, agent_name: str) -> Optional[RemoteAgentClient]:
        """Find a client whose AgentCard name matches `agent_name`."""
        logger.info(f"Looking for client with agent name: {agent_name}")
        for c in self.clients.values():
            if c.agent_card and c.agent_card.name == agent_name:
                logger.info(f"Found client for agent: {agent_name} at {c.base_url}")
                return c
        logger.warning(f"No client found for agent name: {agent_name}")
        return None

    def send_task(self, agent_name: str, message: str) -> str:
        """
        Actually send the user's request to the remote agent via tasks/send JSON-RPC.
        Returns a textual summary or error message.
        """
        logger.info(f"Sending task to agent: {agent_name} with message: {message}")
        client = self.get_client_by_name(agent_name)
        if not client or not client.agent_card:
            logger.error(f"No agent card found for '{agent_name}'")
            return f"Error: No agent card found for '{agent_name}'."

        task_id = str(uuid.uuid4())
        session_id = "session-xyz"
        logger.info(f"Created task {task_id} with session {session_id}")

        try:
            result = client.send_task(task_id, session_id, message)
            # Check final state
            state = result.get("status", {}).get("state", "unknown")
            logger.info(f"Task {task_id} ended with state: {state}")
            
            if state == TaskState.COMPLETED:
                logger.info(f"Task {task_id} completed successfully")
                return f"Task {task_id} completed with message: {result}"
            elif state == TaskState.INPUT_REQUIRED:
                logger.info(f"Task {task_id} requires additional input")
                return f"Task {task_id} needs more input: {result}"
            else:
                logger.warning(f"Task {task_id} ended with unexpected state: {state}")
                return f"Task {task_id} ended with state={state}, result={result}"
        except Exception as exc:
            logger.error(f"Failed to send task to remote agent: {exc}", exc_info=True)
            return f"Remote agent call failed: {exc}"


def make_list_agents_tool(host_agent: HostAgent):
    """Return a synchronous tool function that calls host_agent.list_agents_info()."""

    @tool
    def list_remote_agents_tool() -> list:
        """List available remote agents (name, url, streaming)."""
        logger.info("Tool called: list_remote_agents_tool")
        agents = host_agent.list_agents_info()
        logger.info(f"Found {len(agents)} remote agents")
        return agents

    return list_remote_agents_tool


def make_send_task_tool(host_agent: HostAgent):
    """Return a synchronous tool function that calls host_agent.send_task(...)."""

    @tool
    def send_task_tool(agent_name: str, message: str) -> str:
        """
        Synchronous tool: sends 'message' to 'agent_name'
        via JSON-RPC and returns the result.
        """
        logger.info(f"Tool called: send_task_tool with agent: {agent_name}, message: {message}")
        result = host_agent.send_task(agent_name, message)
        logger.info(f"send_task_tool completed with result length: {len(str(result))}")
        return result

    return send_task_tool


def build_react_agent(host_agent: HostAgent):
    logger.info("Building React agent with host_agent tools")
    
    validate_azure_openai_credentials()
    
    logger.info("Initializing Azure ChatOpenAI model")
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o"),
        temperature=0.1,
        max_tokens=1000
    )
    
    memory = MemorySaver()

    list_tool = make_list_agents_tool(host_agent)
    send_tool = make_send_task_tool(host_agent)

    system_prompt = """
You are a Host Agent that delegates requests to known remote agents.
You have two tools:
1) list_remote_agents_tool(): Lists the remote agents (their name, URL, streaming).
2) send_task_tool(agent_name, message): Sends a text request to the agent.

If the user wants currency conversion, call 'send_task_tool("Currency Agent", "1 USD to GBP")'.
If the user wants weather info, call 'send_task_tool("Weather Agent", "Weather in New York")'.

Always first check what agents are available using list_remote_agents_tool() if you're unsure.
Return the final result to the user in a clear and helpful format.
"""

    agent = create_react_agent(
        model=llm,
        tools=[list_tool, send_tool],
        checkpointer=memory,
        prompt=system_prompt,
    )
    logger.info("React agent created successfully")
    return agent


app = typer.Typer()


@app.command()
def run_agent(remote_url: str = "http://localhost:8000"):
    """
    Start a synchronous HostAgent pointing at 'remote_url'
    and run a simple conversation loop.
    """
    logger.info(f"Starting run_agent with remote_url: {remote_url}")
    # 1) Build the HostAgent
    host_agent = HostAgent([remote_url])

    logger.info("Initializing host_agent")
    host_agent.initialize()
    logger.info("Building React agent")
    react_agent = build_react_agent(host_agent)

    typer.echo(f"Host agent ready. Connected to: {remote_url}")
    typer.echo("Type 'quit' or 'exit' to stop.")
    logger.info("Host agent ready for user input")

    while True:
        user_msg = typer.prompt("\nUser")
        if user_msg.strip().lower() in ["quit", "exit", "bye"]:
            logger.info("User requested to exit")
            typer.echo("Goodbye!")
            break

        logger.info(f"Received user message: {user_msg}")
        logger.info("Invoking React agent")
        raw_result = react_agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config={"configurable": {"thread_id": "cli-session"}},
        )
        logger.info("React agent invocation completed")

        final_text = None

        # If 'raw_result' is a dictionary with "messages", try to find the last AIMessage
        if isinstance(raw_result, dict) and "messages" in raw_result:
            logger.info("Processing messages from raw_result")
            all_msgs = raw_result["messages"]
            for msg in reversed(all_msgs):
                if isinstance(msg, AIMessage):
                    logger.info("Found AIMessage in result")
                    final_text = msg.content
                    break
        else:
            # Otherwise, it's likely a plain string
            if isinstance(raw_result, str):
                logger.info("raw_result is a string")
                final_text = raw_result
            else:
                # fallback: convert whatever it is to string
                logger.info(f"Converting raw_result type {type(raw_result)} to string")
                final_text = str(raw_result)

        # Now print only the final AIMessage content
        logger.info("Displaying response to user")
        typer.echo(f"HostAgent: {final_text}")


# Updated validation function for Azure OpenAI
def validate_azure_openai_credentials():
    """Validate that all required Azure OpenAI environment variables are set."""
    logger.info("Validating Azure OpenAI credentials")
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing Azure OpenAI environment variables: {missing_vars}")
        raise ValueError(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
    
    logger.info("Azure OpenAI credentials validated successfully")

def main():
    """
    Entry point for 'python host_agent.py run-agent --remote-url http://whatever'
    """
    logger.info("Starting host_agent.py main function")
    try:
        validate_azure_openai_credentials()
        app()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Error: {e}")
        typer.echo("Please set the required Azure OpenAI environment variables in your .env file:")
        typer.echo("- AZURE_OPENAI_ENDPOINT")
        typer.echo("- AZURE_OPENAI_API_KEY") 
        typer.echo("- AZURE_OPENAI_DEPLOYMENT_NAME")
        typer.echo("- AZURE_OPENAI_API_VERSION (optional, defaults to 2024-02-01)")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting host_agent.py")
    main()
