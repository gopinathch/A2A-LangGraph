import uuid
from typing import List, Optional

import requests
import typer
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


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
        self.base_url = base_url
        self.agent_card: Optional[AgentCard] = None

    def fetch_agent_card(self) -> AgentCard:
        """GET /.well-known/agent.json to retrieve the remote agent's card."""
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
        return card

    def send_task(self, task_id: str, session_id: str, message_text: str) -> dict:
        """POST / with JSON-RPC request: method=tasks/send."""
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
        r = requests.post(self.base_url, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp and resp["error"] is not None:
            raise RuntimeError(f"Remote agent error: {resp['error']}")
        return resp.get("result", {})


class HostAgent:
    """Holds references to multiple RemoteAgentClients, one per address."""

    def __init__(self, remote_addresses: List[str]):
        self.clients = {}
        for addr in remote_addresses:
            self.clients[addr] = RemoteAgentClient(addr)

    def initialize(self):
        """Fetch agent cards for all addresses (synchronously)."""
        for addr, client in self.clients.items():
            client.fetch_agent_card()

    def list_agents_info(self) -> list:
        """Return a list of {name, description, url, streaming} for each loaded agent."""
        infos = []
        for addr, c in self.clients.items():
            card = c.agent_card
            if card:
                infos.append(
                    {
                        "name": card.name,
                        "description": card.description,
                        "url": card.url,
                        "streaming": card.capabilities.streaming,
                    }
                )
            else:
                infos.append(
                    {
                        "name": "Unknown",
                        "description": "Not loaded",
                        "url": addr,
                        "streaming": False,
                    }
                )
        return infos

    def get_client_by_name(self, agent_name: str) -> Optional[RemoteAgentClient]:
        """Find a client whose AgentCard name matches `agent_name`."""
        for c in self.clients.values():
            if c.agent_card and c.agent_card.name == agent_name:
                return c
        return None

    def send_task(self, agent_name: str, message: str) -> str:
        """
        Actually send the user's request to the remote agent via tasks/send JSON-RPC.
        Returns a textual summary or error message.
        """
        client = self.get_client_by_name(agent_name)
        if not client or not client.agent_card:
            return f"Error: No agent card found for '{agent_name}'."

        task_id = str(uuid.uuid4())
        session_id = "session-xyz"

        try:
            result = client.send_task(task_id, session_id, message)
            # Check final state
            state = result.get("status", {}).get("state", "unknown")
            if state == TaskState.COMPLETED:
                return f"Task {task_id} completed with message: {result}"
            elif state == TaskState.INPUT_REQUIRED:
                return f"Task {task_id} needs more input: {result}"
            else:
                return f"Task {task_id} ended with state={state}, result={result}"
        except Exception as exc:
            return f"Remote agent call failed: {exc}"


def make_list_agents_tool(host_agent: HostAgent):
    """Return a synchronous tool function that calls host_agent.list_agents_info()."""

    @tool
    def list_remote_agents_tool() -> list:
        """List available remote agents (name, url, streaming)."""
        return host_agent.list_agents_info()

    return list_remote_agents_tool


def make_send_task_tool(host_agent: HostAgent):
    """Return a synchronous tool function that calls host_agent.send_task(...)."""

    @tool
    def send_task_tool(agent_name: str, message: str) -> str:
        """
        Synchronous tool: sends 'message' to 'agent_name'
        via JSON-RPC and returns the result.
        """
        return host_agent.send_task(agent_name, message)

    return send_task_tool


def build_react_agent(host_agent: HostAgent):
    # Create the top-level LLM
    llm = ChatOpenAI(model="gpt-4o")
    memory = MemorySaver()

    # Make the two tools referencing our host_agent
    list_tool = make_list_agents_tool(host_agent)
    send_tool = make_send_task_tool(host_agent)

    system_prompt = """
You are a Host Agent that delegates requests to known remote agents.
You have two tools:
1) list_remote_agents_tool(): Lists the remote agents (their name, URL, streaming).
2) send_task_tool(agent_name, message): Sends a text request to the agent.

If the user wants currency conversion, call 'send_task_tool("some_agent_name", "5 USD to EUR")'.
If the user wants weather info, call 'send_task_tool("some_weather_agent", "Weather in city")'.

Return the final result to the user.
"""

    agent = create_react_agent(
        model=llm,
        tools=[list_tool, send_tool],
        checkpointer=memory,
        prompt=system_prompt,
    )
    return agent


app = typer.Typer()


@app.command()
def run_agent(remote_url: str = "http://localhost:8000"):
    """
    Start a synchronous HostAgent pointing at 'remote_url'
    and run a simple conversation loop.
    """
    # 1) Build the HostAgent
    host_agent = HostAgent([remote_url])

    host_agent.initialize()
    react_agent = build_react_agent(host_agent)

    typer.echo(f"Host agent ready. Connected to: {remote_url}")
    typer.echo("Type 'quit' or 'exit' to stop.")

    while True:
        user_msg = typer.prompt("\nUser")
        if user_msg.strip().lower() in ["quit", "exit", "bye"]:
            typer.echo("Goodbye!")
            break

        raw_result = react_agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config={"configurable": {"thread_id": "cli-session"}},
        )

        final_text = None

        # If 'raw_result' is a dictionary with "messages", try to find the last AIMessage
        if isinstance(raw_result, dict) and "messages" in raw_result:
            all_msgs = raw_result["messages"]
            for msg in reversed(all_msgs):
                if isinstance(msg, AIMessage):
                    final_text = msg.content
                    break
        else:
            # Otherwise, it's likely a plain string
            if isinstance(raw_result, str):
                final_text = raw_result
            else:
                # fallback: convert whatever it is to string
                final_text = str(raw_result)

        # Now print only the final AIMessage content
        typer.echo(f"HostAgent: {final_text}")


def main():
    """
    Entry point for 'python sync_host_agent_cli.py run-agent --remote-url http://whatever'
    """
    app()


if __name__ == "__main__":
    main()
