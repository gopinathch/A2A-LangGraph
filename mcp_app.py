import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing FastMCP server")
mcp = FastMCP(name="MinimalServer", host="0.0.0.0", port=3000)
logger.info("FastMCP server initialized")


@mcp.tool()
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):

    logger.info(f"get_exchange_rate called with currency_from={currency_from}, currency_to={currency_to}, currency_date={currency_date}")
    
    result = {
        "amount": 1,
        "base": currency_from,
        "date": currency_date,
        "rates": {currency_to: 0.85},  # Beispiel-Rate
    }
    
    logger.info(f"get_exchange_rate returning: {result}")
    return result


if __name__ == "__main__":
    logger.info("Starting MCP server with SSE transport")
    mcp.run(transport="sse")
    logger.info("MCP server stopped")
