import json
import logging
from typing import Any, Dict, List

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry, truncate: bool = False) -> str:
        """
        Format an entry for display in the search results.
        :param entry: The entry to format.
        :param truncate: Whether to truncate the content to 200 characters.
        :return: The formatted entry.
        """
        content = entry.content
        if truncate and len(content) > 200:
            content = content[:200] + "..."
            
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            assert self.qdrant_settings.collection_name is not None
            return await store(
                ctx, information, self.qdrant_settings.collection_name, metadata
            )

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
        ) -> List[str]:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :return: A list of entries found with truncated content to save context.
            """
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
            )
            if not entries:
                return [f"No information found for the query '{query}'"]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry, truncate=True))
            return content

        async def find_with_default_collection(
            ctx: Context,
            query: str,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            return await find(ctx, query, self.qdrant_settings.collection_name)

        async def list_collections(
            ctx: Context,
        ) -> List[str]:
            """
            List all available collections in the Qdrant server.
            :param ctx: The context for the request.
            :return: A list of collection names.
            """
            await ctx.debug("Listing collections in Qdrant")
            collection_names = await self.qdrant_connector.get_collection_names()
            if not collection_names:
                return ["No collections found in Qdrant server."]
            return [f"Available collections: {', '.join(collection_names)}"]

        async def get_collection_info(
            ctx: Context,
            collection_name: str,
        ) -> Dict[str, Any]:
            """
            Get detailed information about a specific collection.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection to get information for.
            :return: A dictionary containing collection information.
            """
            await ctx.debug(f"Getting information for collection {collection_name}")
            collection_info = await self.qdrant_connector.get_collection_info(collection_name)
            if not collection_info:
                return {"error": f"Collection '{collection_name}' not found."}
            return collection_info

        # Register the tools depending on the configuration

        if self.qdrant_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )
        else:
            self.add_tool(
                find,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database

            if self.qdrant_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
            else:
                self.add_tool(
                    store,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
                
        async def match(
            ctx: Context,
            metadata: Metadata,
            collection_name: str,
        ) -> List[str]:
            """
            Find memories in Qdrant that exactly match the provided metadata.
            :param ctx: The context for the request.
            :param metadata: The metadata to match against.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                   the default collection is used.
            :return: A list of entries found.
            """
            await ctx.debug(f"Matching results for metadata {metadata}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )

            entries = await self.qdrant_connector.search_by_metadata(
                metadata,
                collection_name=collection_name,
            )
            if not entries:
                return [f"No information found for the metadata '{metadata}'"]
            content = [
                f"Results for the metadata match '{metadata}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry, truncate=False))
            return content

        async def match_with_default_collection(
            ctx: Context,
            metadata: Metadata,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            return await match(ctx, metadata, self.qdrant_settings.collection_name)

        # Always add these tools for collection management
        self.add_tool(
            list_collections,
            name="qdrant-list-collections",
            description="List all available collections in the Qdrant server.",
        )
        
        self.add_tool(
            get_collection_info,
            name="qdrant-collection-info",
            description="Get detailed information about a specific collection, including its configuration and schema.",
        )
        
        # Add the match tool
        if self.qdrant_settings.collection_name:
            self.add_tool(
                match_with_default_collection,
                name="qdrant-match",
                description=self.tool_settings.tool_match_description,
            )
        else:
            self.add_tool(
                match,
                name="qdrant-match",
                description=self.tool_settings.tool_match_description,
            )
