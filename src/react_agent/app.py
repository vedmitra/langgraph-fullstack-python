from fasthtml.common import *
import uuid
from langgraph_sdk import get_client
from starlette.responses import StreamingResponse
from datetime import datetime
from typing import TypedDict, Dict, List

# Initialize the LangGraph client
langgraph_client = get_client()

# Define HTML headers for styling and client-side functionality
hdrs = (
    picolink,
    Script(src="https://cdn.tailwindcss.com"),
    Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
    ),
    Script(src="https://unpkg.com/htmx.org@1.9.2/dist/htmx.min.js"),
    Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
)

# Create the FastHTML app with headers and default styling
app = FastHTML(
    hdrs=hdrs, ct_hdr=True, cls="flex flex-row h-screen", live=True, debug=True
)


# Define Thread type for clarity (though not strictly necessary in Python)
class Thread(TypedDict):
    thread_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict
    status: str
    values: dict
    interrupts: Dict[str, List[dict]]


# Function to get or set user_id via cookie
def get_user_id(request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    return user_id


# Function to render a chat message
def ChatMessage(msg, idx):
    bubble_class = (
        "chat-bubble-primary" if msg["type"] == "human" else "chat-bubble-secondary"
    )
    chat_class = "chat-end" if msg["type"] == "human" else "chat-start"
    return Div(
        Div(msg["type"], cls="chat-header"),
        Div(
            msg["content"], id=f"chat-content-{idx}", cls=f"chat-bubble {bubble_class}"
        ),
        id=f"chat-message-{idx}",
        cls=f"chat {chat_class}",
    )


# Function to create the chat input field
def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
    )


# Sidebar with conversation list
async def ConversationList(user_id: str, current_thread_id: str):
    threads = await langgraph_client.threads.search(
        metadata={"user_id": user_id}, limit=50, offset=0
    )
    return Div(
        H2("Conversations", cls="text-lg font-bold mb-2"),
        Div(
            *[
                A(
                    f"Thread {i+1} ({thread['created_at']})",
                    href=f"/conversations/{thread['thread_id']}",
                    cls="block p-2 hover:bg-gray-200 rounded"
                    + (
                        " bg-gray-300"
                        if thread["thread_id"] == current_thread_id
                        else ""
                    ),
                )
                for i, thread in enumerate(threads)
            ],
            cls="overflow-y-auto h-[calc(100vh-4rem)]",
        ),
        cls="w-1/4 bg-gray-100 p-4 border-r",
    )


# Root route: Redirect to a new conversation
@app.route("/")
def get(request):
    user_id = get_user_id(request)
    thread_id = str(uuid.uuid4())
    return Redirect(
        f"/conversations/{thread_id}",
        cookies=[("user_id", user_id, {"httponly": True, "max_age": 31536000})],
    )


# Conversation route: Display the chat interface with sidebar
@app.route("/conversations/{thread_id}")
async def get(thread_id: str, request):
    user_id = get_user_id(request)

    # Create thread with user_id in metadata
    await langgraph_client.threads.create(
        thread_id=thread_id, if_exists="do_nothing", metadata={"user_id": user_id}
    )

    # Fetch thread state
    try:
        state = await langgraph_client.threads.get_state(thread_id)
        messages = state["values"]["messages"]
    except Exception:
        messages = []

    # New Thread button
    new_thread_button = Button(
        "New Thread",
        cls="btn btn-secondary btn-sm mr-2",
        hx_get="/new-thread",
        hx_swap="none",
    )

    # Main chat content
    chat_content = Div(
        Div(
            Titled("Chatbot Demo", ""),
            new_thread_button,
            cls="flex justify-between items-center mb-4",
        ),
        Div(
            *[ChatMessage(msg, i) for i, msg in enumerate(messages)],
            id="chatlist",
            cls="chat-box h-[calc(100vh-8rem)] overflow-y-auto",
        ),
        Form(
            Group(ChatInput(), Button("Send", cls="btn btn-primary")),
            hx_post=f"/conversations/{thread_id}/send-message",
            hx_target="#chatlist",
            hx_swap="beforeend",
            cls="flex space-x-2 mt-2",
        ),
        cls="flex-1 p-4",
    )

    # Full page with sidebar and chat
    page = Div(
        await ConversationList(user_id, thread_id),
        chat_content,
        cls="flex w-full h-full",
    )

    return Titled(
        "Chatbot Demo",
        page,
        cookies=[("user_id", user_id, {"httponly": True, "max_age": 31536000})],
    )


# Route for creating a new thread
@app.get("/new-thread")
def new_thread(request):
    thread_id = str(uuid.uuid4())
    return Redirect(f"/conversations/{thread_id}")


# Function to create an assistant message placeholder with SSE
def AssistantMessagePlaceholder(thread_id, msg):
    content_id = f"assistant-content-{uuid.uuid4()}"
    return Div(
        Div("", id=content_id, cls="chat-bubble chat-bubble-secondary"),
        cls="chat chat-start",
        hx_ext="sse",
        sse_connect=f"/conversations/{thread_id}/get-message?msg={msg}",
        sse_swap="message",
        hx_target=f"#{content_id}",
        hx_swap="beforeend",
    )


# Route to handle sending a new message
@app.post("/conversations/{thread_id}/send-message")
def send_message(thread_id: str, msg: str):
    user_msg_div = ChatMessage(
        {"type": "human", "content": msg}, f"user-{uuid.uuid4()}"
    )
    assistant_placeholder = AssistantMessagePlaceholder(thread_id, msg)
    return user_msg_div, assistant_placeholder, ChatInput()


# Generator for streaming assistant responses
async def message_generator(thread_id: str, msg: str):
    async for event in langgraph_client.runs.stream(
        thread_id,
        "agent",
        input={"messages": [{"type": "human", "content": msg}]},
        stream_mode="messages-tuple",
        multitask_strategy="interrupt",
    ):
        if event.event == "messages":
            for chunk_msg in event.data:
                chunk = chunk_msg.get("content", "")
                if chunk.strip():
                    yield f"event: message\ndata: {chunk}\n\n"
    yield "event: close\ndata:\n\n"


# Route to stream assistant responses via SSE
@app.get("/conversations/{thread_id}/get-message")
async def get_message(thread_id: str, msg: str):
    return StreamingResponse(
        message_generator(thread_id, msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
