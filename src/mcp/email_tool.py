import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_email(recipient_email: str, subject: str, body: str) -> str:
    """
    Sends an email using standard SMTP.
    Requires SMTP_SERVER, SMTP_PORT, EMAIL_SENDER, and EMAIL_PASSWORD in .env.
    """
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    
    if not sender or not password:
        return "Error: EMAIL_SENDER or EMAIL_PASSWORD missing in .env. Cannot send email."
        
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        
        return f"Success: Email sent to {recipient_email}."
    except Exception as e:
        return f"Error: Failed to send email. {str(e)}"
