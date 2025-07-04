import logging
import os

import click
from dotenv import load_dotenv

from agent import CurrencyAgent
from custom_types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
from push_notification_auth import PushNotificationSenderAuth
from server import A2AServer
from task_manager import AgentTaskManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        # Check for Azure OpenAI required environment variables
        required_azure_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY", 
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]
        
        missing_vars = [var for var in required_azure_vars if not os.getenv(var)]
        if missing_vars:
            raise MissingAPIKeyError(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
        skill = AgentSkill(
            id="convert_currency",
            name="Currency Exchange Rates Tool",
            description="Helps with exchange values between various currencies",
            tags=["currency conversion", "currency exchange"],
            examples=["What is exchange rate between USD and GBP?"],
        )
        agent_card = AgentCard(
            name="Currency Agent",
            description="Helps with exchange rates for currencies",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=CurrencyAgent(), notification_sender_auth=notification_sender_auth
            ),
            host=host,
            port=port,
        )

        server.app.add_route(
            "/.well-known/jwks.json",
            notification_sender_auth.handle_jwks_endpoint,
            methods=["GET"],
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
