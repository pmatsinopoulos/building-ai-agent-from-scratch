from tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. This action cannot be undone."""

    # Mock implementation: in a real system this would call an SMTP/API client.
    return (
        f"Email sent to {to}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}"
    )
