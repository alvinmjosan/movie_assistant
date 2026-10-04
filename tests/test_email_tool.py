from unittest.mock import patch

from mcp.email_tool import send_email


def test_send_email_invalid_recipient():
    out = send_email("not-an-email", "s", "b")
    assert out.startswith("Error")


def test_send_email_missing_credentials(monkeypatch):
    monkeypatch.delenv("EMAIL_SENDER", raising=False)
    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    out = send_email("a@b.com", "s", "b")
    assert "Error" in out


def test_send_email_success(monkeypatch):
    monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")

    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, *a): pass

    with patch("mcp.email_tool.smtplib.SMTP", FakeSMTP):
        out = send_email("to@example.com", "subj", "body")
    assert out.startswith("Success")