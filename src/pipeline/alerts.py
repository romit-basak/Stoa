"""Email the team from inside a task: data warnings (validate, osm_watch) and training results
(src/models/flow/notify.py).

Recipients are STOA_ALERT_EMAILS (comma-separated), the server is the smtp_default connection
docker-compose.yaml passes in AIRFLOW_CONN_SMTP_DEFAULT, and the sender is STOA_SMTP_USER. The
settings are read directly rather than through Airflow: inside a task, Airflow 3 hides the
metadata database from the task's process, so importing airflow there fails.

Nothing here raises. An email is a report, never a reason to fail a task, so a failed send is
only logged.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any

log = logging.getLogger("stoa.pipeline")


def recipients() -> list[str]:
    return [
        e.strip()
        for e in os.environ.get("STOA_ALERT_EMAILS", "").split(",")
        if e.strip()
    ]


def smtp_settings() -> dict[str, Any] | None:
    """The smtp_default connection (JSON), the one the failure emails use, or None if it isn't
    set or has no login."""
    raw = os.environ.get("AIRFLOW_CONN_SMTP_DEFAULT", "")
    try:
        conn = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        log.warning("AIRFLOW_CONN_SMTP_DEFAULT is not JSON, so no email can be sent")
        return None
    if not conn.get("host") or not conn.get("login"):
        return None
    return conn


def send_email(subject: str, html: str, what: str = "the alert") -> bool:
    """Send an HTML email to STOA_ALERT_EMAILS. Returns whether it went out; `what` names it
    in the log ("the warnings", "the training result")."""
    try:
        to = recipients()
        if not to:
            log.info("STOA_ALERT_EMAILS is empty, so %s was not emailed", what)
            return False
        conn = smtp_settings()
        if conn is None:
            log.info(
                "no SMTP settings (AIRFLOW_CONN_SMTP_DEFAULT), so %s was not emailed",
                what,
            )
            return False
        mail = EmailMessage()
        mail["Subject"] = subject
        mail["From"] = os.environ.get("STOA_SMTP_USER") or conn["login"]
        mail["To"] = ", ".join(to)
        mail.set_content("This email needs an HTML-capable mail client.")
        mail.add_alternative(html, subtype="html")
        with smtplib.SMTP(
            conn["host"], int(conn.get("port") or 587), timeout=30
        ) as smtp:
            # Gmail on 587: plain connection upgraded with STARTTLS (disable_ssl in extra).
            smtp.starttls()
            smtp.login(conn["login"], conn.get("password") or "")
            smtp.send_message(mail)
    except Exception:
        log.exception("could not email %s", what)
        return False
    log.info("emailed %s to %s", what, ", ".join(to))
    return True
