from google.cloud import secretmanager

from google.cloud import secretmanager
import os

def access_secret_version(secret_id: str, version_id: str = "latest") -> str:
    """
    Accesses a secret version from Google Secret Manager.

    Args:
        secret_id (str): The ID of the secret to retrieve.
        version_id (str): The version of the secret to retrieve. Defaults to "latest".

    Returns:
        str: The decoded secret payload.

    Raises:
        Exception: If there is an error accessing the secret version.
    """
    try:
        # Get the project ID from an environment variable
        project_id = os.getenv("GCP_PROJECT_ID", "467851153648")  # Replace with your default project ID

        # Initialize the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

        # Access the secret version
        response = client.access_secret_version(name=name)

        # Decode and return the secret payload
        return response.payload.data.decode("UTF-8")

    except Exception as e:
        # Log or handle the exception as needed
        raise Exception(f"Failed to access secret version: {e}")
