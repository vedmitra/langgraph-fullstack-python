# mypy: disable-error-code="no-untyped-def,misc"

"""FastHTML-based chat interface for the LangGraph agent.

This module implements a real-time chat interface using FastHTML components and Server-Sent Events (SSE)
for streaming responses. It maintains conversation history and supports multiple chat threads per user.

Mostly based on: https://github.com/AnswerDotAI/fasthtml-example/blob/main/04_sse/sse_chatbot.py
"""

import uuid
from typing import AsyncGenerator, Dict

from fasthtml.common import (  # type: ignore
    H2,
    A,
    Button,
    Div,
    FastHTML,
    Form,
    Group,
    Input,
    Link,
    Script,
    Title,
    Titled,
    picolink,
)
from fasthtml.core import Request  # type: ignore
from langgraph_sdk import get_client
from starlette.responses import RedirectResponse, StreamingResponse

# Initialize the LangGraph client
langgraph_client = get_client()

# Define HTML headers for styling and client-side functionality
tlink = (Script(src="https://cdn.tailwindcss.com"),)
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
app = FastHTML(hdrs=(tlink, dlink, picolink, sselink), live=True)


def get_user_id(request: Request) -> str:
    """Get or create a user ID from cookies.

    Returns a UUID if no user ID cookie exists.
    """
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = uuid.uuid4()
    return str(user_id)


def ChatMessage(msg: Dict[str, str], idx: str | int) -> Div:
    """Render a chat message bubble.

    Creates a styled message bubble with different colors for user/assistant messages.
    """
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


def ChatInput() -> Input:
    """Create the message input field.

    Returns a text input configured for HTMX swapping.
    """
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
    )


async def ConversationList(user_id: str, current_thread_id: str) -> Div:
    """Render the sidebar list of conversations.

    Shows all threads for the user with the current thread highlighted.
    """
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


@app.route("/")  # type: ignore
async def root(request: Request):
    """Root index for redirecting to a new conversation."""
    thread_id = str(uuid.uuid4())
    return RedirectResponse(f"/conversations/{thread_id}", status_code=302)


@app.get("/conversations/{thread_id}")  # type: ignore[misc]
async def conversation(thread_id: str, request: Request):
    """Display the chat interface for a specific conversation.

    Shows message history and handles new message streaming.
    """
    user_id = get_user_id(request)

    # Create thread with user_id in metadata
    await langgraph_client.threads.create(
        thread_id=thread_id, if_exists="do_nothing", metadata={"user_id": user_id}
    )

    # Fetch thread state
    try:
        state = await langgraph_client.threads.get_state(thread_id)
        values = state["values"]
        if isinstance(values, list):
            messages = values[-1]["messages"]
        else:
            messages = values["messages"]
    except Exception:
        messages = []

    # Define the "New Thread" button as a regular link
    new_thread_button = A(
        "New Thread",
        href="/new-thread",
        cls="btn btn-secondary btn-sm mr-2",
    )

    # Main chat content
    chat_content = Div(
        Div(
            Titled("Chatbot Demo. Do not share private data - this is an unauthenticated demo!", ""),
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

    page = Div(
        await ConversationList(user_id, thread_id),
        chat_content,
        cls="flex w-full h-full",
    )
    return (
        Title(
            "Chatbot Demo",
        ),
        page,
    )


@app.get("/new-thread")  # type: ignore[misc]
async def new_thread(request: Request):
    """Create a new conversation thread.

    Redirects to the new thread's conversation page.
    """
    thread_id = str(uuid.uuid4())
    return RedirectResponse(f"/conversations/{thread_id}", status_code=302)


def AssistantMessagePlaceholder(thread_id: str, msg: str) -> Div:
    """Create a placeholder for streaming assistant responses.

    Sets up SSE connection for real-time message updates.
    """
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


@app.post("/conversations/{thread_id}/send-message")  # type: ignore[misc]
async def send_message(thread_id: str, msg: str):
    """Handle sending a new message in a conversation.

    Returns the user message, assistant placeholder, and new input field.
    """
    user_msg_div = ChatMessage(
        {"type": "human", "content": msg}, f"user-{uuid.uuid4()}"
    )
    assistant_placeholder = AssistantMessagePlaceholder(thread_id, msg)
    return user_msg_div, assistant_placeholder, ChatInput()


async def message_generator(thread_id: str, msg: str) -> AsyncGenerator[str, None]:
    """Stream assistant responses via SSE.

    Yields message chunks as they are received from the LangGraph agent.
    """
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
@app.get("/conversations/{thread_id}/get-message")  # type: ignore[misc]
async def get_message(thread_id: str, msg: str):
    """SSE endpoint for streaming assistant responses.

    Sets up proper headers for SSE streaming.
    """
    return StreamingResponse(
        message_generator(thread_id, msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
