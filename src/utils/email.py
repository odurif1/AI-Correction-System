"""
Email service for La Corrigeuse.

Sends transactional emails via SendGrid (password reset, notifications).
"""

import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from loguru import logger
from config.settings import get_settings


class EmailService:
    """SendGrid email service wrapper."""

    def __init__(self):
        self.settings = get_settings()
        self.client = sendgrid.SendGridAPIClient(api_key=self.settings.sendgrid_api_key)

    async def send_password_reset(
        self,
        to_email: str,
        reset_link: str,
        user_name: str = None
    ) -> bool:
        """
        Send password reset email.

        Args:
            to_email: Recipient email address
            reset_link: Password reset link with token
            user_name: Optional user name for personalization

        Returns:
            True if email sent successfully, False otherwise
        """
        if self.settings.sendgrid_sandbox_mode:
            logger.info(f"[SANDBOX] Would send password reset email to {to_email}: {reset_link}")
            return True

        message = Mail(
            from_email=Email(self.settings.sendgrid_sender, "La Corrigeuse"),
            to_emails=To(to_email),
            subject="Réinitialisation de votre mot de passe - La Corrigeuse",
            plain_text_content=self._render_reset_email(reset_link, user_name)
        )

        try:
            response = self.client.send(message)
            logger.info(f"Password reset email sent to {to_email}, status: {response.status_code}")
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Failed to send password reset email: {e}")
            return False

    def _render_reset_email(self, reset_link: str, user_name: str) -> str:
        """Render plain text password reset email."""
        greeting = f"Bonjour {user_name}," if user_name else "Bonjour,"
        return f"""{greeting}

Vous avez demandé la réinitialisation de votre mot de passe sur La Corrigeuse.

Cliquez sur le lien ci-dessous pour définir un nouveau mot de passe :
{reset_link}

Ce lien expire dans 30 minutes.

Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.

---
L'équipe La Corrigeuse
"""
