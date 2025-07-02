import asyncio
import logging
import os
from typing import Any, AsyncIterable, Dict, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Configure logger
logger = logging.getLogger(__name__)




memory = MemorySaver()


def _fetch_mcp_tools_sync() -> list:
    """
    Helper function: creates simple sync wrappers that establish fresh MCP connections.
    """
    logger.info("Starting _fetch_mcp_tools_sync")
    
    servers_config = {
        "currency_server": {
            "transport": "sse", 
            "url": "http://127.0.0.1:3000/sse",
        }
    }
    
    async def _get_tools_schema():
        async with MultiServerMCPClient(servers_config) as client:
            tools = client.get_tools()
            return [(tool.name, tool.description, tool.args_schema) for tool in tools]
    
    # Get tool schemas
    tools_schema = asyncio.run(_get_tools_schema())
    logger.info(f"Retrieved {len(tools_schema)} tool schemas")
    
    sync_tools = []
    for name, description, args_schema in tools_schema:
        logger.info(f"Creating sync wrapper for tool: {name}")
        
        def create_tool_wrapper(tool_name):
            def sync_tool_func(**kwargs):
                logger.info(f"Executing {tool_name} with args: {kwargs}")
                
                async def _execute_tool():
                    async with MultiServerMCPClient(servers_config) as client:
                        tools = client.get_tools()
                        for tool in tools:
                            if tool.name == tool_name:
                                return await tool.ainvoke(kwargs)
                        raise ValueError(f"Tool {tool_name} not found")
                
                try:
                    result = asyncio.run(_execute_tool())
                    logger.info(f"Tool {tool_name} returned: {result}")
                    return result
                except Exception as e:
                    logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
                    raise e
            
            return sync_tool_func
        
        sync_tool = StructuredTool.from_function(
            func=create_tool_wrapper(name),
            name=name,
            description=description,
            args_schema=args_schema
        )
        
        sync_tools.append(sync_tool)
    
    logger.info(f"Created {len(sync_tools)} sync tools")
    return sync_tools


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class CurrencyAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for currency conversions. "
        "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
        "If the user asks about anything other than currency conversion or exchange rates, "
        "politely state that you cannot help with that topic and can only assist with currency-related queries. "
        "Do not attempt to answer unrelated questions or use tools for other purposes."
        "Set response status to input_required if the user needs to provide more information."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )

    def __init__(self):
        logger.info("Initializing CurrencyAgent")
        
        # Validate Azure OpenAI environment variables
        required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
        
        # Fetch MCP tools
        logger.info("Fetching MCP tools")
        self.tools = _fetch_mcp_tools_sync()
        logger.info(f"Successfully fetched {len(self.tools) if self.tools else 0} MCP tools")

        # Initialize Azure OpenAI model using environment variables
        logger.info("Initializing Azure ChatOpenAI model")
        self.model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),  # <-- Correct deployment name
            model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o"),  # <-- Correct model name
    )
        
        logger.info("Creating react agent graph")
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )
        logger.info("CurrencyAgent initialization completed")

    def invoke(self, query, sessionId) -> str:
        logger.info(f"CurrencyAgent.invoke called with query: '{query}', sessionId: '{sessionId}'")
        config = {"configurable": {"thread_id": sessionId}}
        logger.info(f"Created config: {config}")
        
        logger.info("Invoking graph with user message")
        self.graph.invoke({"messages": [("user", query)]}, config)
        logger.info("Graph invocation completed, getting agent response")
        
        result = self.get_agent_response(config)
        logger.info(f"CurrencyAgent.invoke completed with result: {result}")
        return result

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        logger.info(f"CurrencyAgent.stream called with query: '{query}', sessionId: '{sessionId}'")
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}
        logger.info(f"Created inputs: {inputs}, config: {config}")

        logger.info("Starting graph stream")
        for item in self.graph.stream(inputs, config, stream_mode="values"):
            logger.info(f"Processing stream item: {type(item)}")
            message = item["messages"][-1]
            logger.info(f"Message type: {type(message)}, content preview: {str(message)[:100]}...")
            
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                logger.info(f"AIMessage with {len(message.tool_calls)} tool calls detected")
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }
            elif isinstance(message, ToolMessage):
                logger.info("ToolMessage detected")
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }

        logger.info("Graph stream completed, getting final agent response")
        final_response = self.get_agent_response(config)
        logger.info(f"Final response: {final_response}")
        yield final_response

    def get_agent_response(self, config):
        logger.info("get_agent_response called")
        print("Fetching agent response...")
        
        logger.info("Getting current state from graph")
        current_state = self.graph.get_state(config)
        logger.info(f"Current state retrieved: {current_state}")
        
        structured_response = current_state.values.get("structured_response")
        logger.info(f"Structured response: {structured_response}, type: {type(structured_response)}")
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            logger.info(f"Valid ResponseFormat found with status: {structured_response.status}")
            if structured_response.status == "input_required":
                logger.info("Returning input_required response")
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "error":
                logger.info("Returning error response")
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "completed":
                logger.info("Returning completed response")
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        logger.info("No valid structured response found, returning default error response")
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
