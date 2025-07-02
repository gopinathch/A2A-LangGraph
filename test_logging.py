#!/usr/bin/env python3
"""
Test script to verify logging is working in all components.
This script will help you debug the MCP connection and agent invocation.
"""

import logging
import asyncio
import sys
from agent import CurrencyAgent

# Configure logging to see all log levels
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_currency_agent():
    """Test the CurrencyAgent to see if MCP connection works."""
    logger.info("=== Starting Currency Agent Test ===")
    
    try:
        # Initialize the agent
        logger.info("Creating CurrencyAgent instance...")
        agent = CurrencyAgent()
        logger.info("CurrencyAgent created successfully")
        
        # Test a simple query
        query = "What is the exchange rate from USD to EUR?"
        session_id = "test-session-123"
        
        logger.info(f"Testing invoke method with query: '{query}'")
        result = agent.invoke(query, session_id)
        logger.info(f"Invoke result: {result}")
        
        logger.info("Testing stream method...")
        async for item in agent.stream(query, session_id):
            logger.info(f"Stream item: {item}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return False
    
    logger.info("=== Currency Agent Test Completed ===")
    return True

if __name__ == "__main__":
    logger.info("Starting logging test script")
    asyncio.run(test_currency_agent())
    logger.info("Test script completed")
