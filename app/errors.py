"""
Custom exception for handling API errors
"""

class UpstreamServiceError(Exception):
    def __init__(self, service: str, message: str = "An error occurred with the upstream service"):
        """
        Exception raised for errors in upstream services.
        Attributes:
            service (str): Name of the upstream service.
            message (str): Explanation of the error.
        """
        self.service = service
        self.message = message
        super().__init__(message)