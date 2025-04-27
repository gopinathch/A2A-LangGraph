from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="MinimalServer", host="0.0.0.0", port=3000)


@mcp.tool()
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """Dummy-Tool, das 1:1 eine statische Antwort zurückgibt, anstelle eines echten API-Calls.

    Args:
        currency_from: Die Quellwährung (z.B. "USD").
        currency_to: Die Zielwährung (z.B. "EUR").
        currency_date: Das Datum für den Wechselkurs oder "latest". Standard "latest".

    Returns:
        Ein Dictionary mit statischen Placeholder-Daten.
    """
    return {
        "amount": 1,
        "base": currency_from,
        "date": currency_date,
        "rates": {currency_to: 0.85},  # Beispiel-Rate
    }


if __name__ == "__main__":
    mcp.run(transport="sse")
