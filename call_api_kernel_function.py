from semantic_kernel.functions import kernel_function, KernelFunction
from typing import Annotated
import os
import requests
import json

from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv("../.env", override=True)
API_SERVICE_URL = os.getenv("API_URL", "http://localhost:8000")

available_apis = {
    "log-storage": "urlToCall",
    "onlineSearch": "urlToCall",
    "jiraSearch": "urlToCall"
}

# Define a simple plugin class with useful functions
class ApiInteractionPlugin:

    @kernel_function(description="Get available APIs that you can call from the orchestrator. The functions returns a json with the API name and URL.")
    def get_available_orchestrator_apis(self) -> str:
        """Get a list of available APIs that can be called from the orchestrator."""
        print("Retrieving available APIs...")
        return available_apis
    
    @kernel_function(description="Call the API")
    def call_orchestrator_api(
        self,
        api_url: Annotated[str, "The URL of the API to call."],
        payload: Annotated[dict, "The payload to send to the API in json format."],
    ) -> str:
        print(f"Calling API: {api_url} with payload: {payload}")
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()  # Raise exception for error codes
            return "Successfully called API: " + api_url + " with response: " + response.text

        except requests.exceptions.RequestException as e:
            print(f"Error calling API: {str(e)}")
            return f"Error calling API: {str(e)}"
