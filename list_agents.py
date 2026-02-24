import asyncio
import os
from azure.ai.agents.aio import AgentsClient
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

async def main():
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    credential = DefaultAzureCredential()
    
    async with AgentsClient(endpoint=endpoint, credential=credential) as client:
        async for agent in client.list_agents():
            print(f"Agent Name: {agent.name}, ID: {agent.id}")
    await credential.close()

asyncio.run(main())
