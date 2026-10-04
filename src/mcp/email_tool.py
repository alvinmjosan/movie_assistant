import os
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def send_email(recipient_email: str, subject: str, body: str) -> str:
    """
    Sends an email using SMTP. Returns a success or error string.
    Interface preserved; validation and error handling improved.
    """
    if not recipient_email or not _EMAIL_RE.match(recipient_email):
        return f"Error: Invalid recipient email: {recipient_email!r}"
    if not subject or not body:
        return "Error: subject and body are required."

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")

    if not sender or not password:
        return "Error: EMAIL_SENDER or EMAIL_PASSWORD missing in .env."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        return f"Success: Email sent to {recipient_email}."
    except Exception as e:
        return f"Error: Failed to send email. {e}"