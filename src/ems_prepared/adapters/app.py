from uuid import UUID, uuid4

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

from ems_prepared.policies.pydantic_graph.emergency_main_graph import loop_graph
from ems_prepared.util.settings import InputMode, Settings

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/session")
async def start_session(user_id: UUID | None = None, session_id: UUID | None = None):
    this_session_id = session_id or uuid4()
    resolved_username: str = (
        user_id.hex if user_id is not None else f"guest-{this_session_id.hex}"
    )
    return {"username": resolved_username, "status": "session_started"}


@app.post("/session/{session_id}")
async def resume_session(user_id: UUID | None = None, session_id: UUID | None = None):
    this_session_id = session_id or uuid4()
    resolved_username = user_id or f"guest-{this_session_id.hex}"
    return {"username": resolved_username, "status": "session_started"}


@app.post("/session/{session_id}/messages")
async def send_message(user_id: UUID | None = None):
    pass


@app.get("/session/{session_id}/state")
async def get_state(session_id: UUID):
    pass


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):
    await websocket.accept()
    try:
        deps = Settings(
            name="WebSocketUser",
            # user_id=UUID(int=0),
            user_id=uuid4(),
            # session_id=UUID(int=0),
            session_id=uuid4(),
            websocket=websocket,
            call_origin=InputMode.API,
        )
        deps.logger.debug(
            f"WebSocket session started: user_id={deps.user_id} session_id={deps.session_id}"
        )

        # await resume_session(deps.user_id, deps.session_id)
        await loop_graph(deps)
    except WebSocketDisconnect:
        print("Client disconnected")
        pass


@app.get("/html")
async def get():
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat</title>
        </head>
        <body>
            <h1>WebSocket Chat</h1>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <ul id='messages'>
            </ul>
            <script>
                var ws = new WebSocket("ws://localhost:8000/ws");
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
                function sendMessage(event) {
                    var input = document.getElementById("messageText")
                    ws.send(input.value)
                    input.value = ''
                    event.preventDefault()
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html)
