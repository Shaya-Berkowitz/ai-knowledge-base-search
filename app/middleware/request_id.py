"""
Middleware to add a unique request ID to each incoming HTTP request.
"""

import uuid
import logging
from fastapi import Request, Response

logger = logging.getLogger(__name__)

async def request_id_middleware(request: Request, call_next) -> Response:
    """
    Middleware to add a unique request ID to each incoming HTTP request.
    Args:
        request (Request): Incoming HTTP request.
        call_next: Function to call the next middleware or endpoint
    Returns:
        Response: HTTP response with ID added to headers."""
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response