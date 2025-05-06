import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv


def setup_logging():
    """Configure logging settings."""
    log_level_name = os.getenv("MCP_LOG_LEVEL", "INFO")
    log_file = os.getenv("MCP_LOG_FILE")
    
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific package loggers
    logging.getLogger("mcp_server_qdrant").setLevel(log_level)
    logging.getLogger("qdrant_client").setLevel(log_level)


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """
    # Load environment variables from .env file if it exists
    env_path = Path(os.getcwd()) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment variables from {env_path}")
        print(f"Using collection: {os.getenv('COLLECTION_NAME', 'Not specified')}")
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting mcp-server-qdrant")

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    args = parser.parse_args()
    
    logger.info(f"Using transport protocol: {args.transport}")

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from mcp_server_qdrant.server import mcp

    logger.info("MCP server initialized, running...")
    mcp.run(transport=args.transport)
