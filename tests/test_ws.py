import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8080/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Send a message
        request = {
            "content": "Tell me a dad joke"
        }
        await websocket.send(json.dumps(request))
        
        # Receive messages
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received: {data}")
                if data.get("type") == "done":
                    break
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

if __name__ == "__main__":
    asyncio.run(test_websocket())
