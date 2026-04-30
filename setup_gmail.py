"""
setup_gmail.py - One-time Gmail OAuth2 authorisation helper.

Run this ONCE before starting the main application:
    python setup_gmail.py

It will open a browser, ask you to sign in with Google and grant permissions,
then save a token.json file that the app reuses automatically.
"""
import sys
from pathlib import Path

from app.config import settings


def main():
    creds_path = Path(settings.gmail_credentials_path)
    if not creds_path.exists():
        print(
            f"\n❌  credentials.json not found at '{creds_path}'.\n\n"
            "Follow these steps to create it:\n"
            "  1. Go to https://console.cloud.google.com/\n"
            "  2. Create a project (or select an existing one).\n"
            "  3. Enable the Gmail API:\n"
            "       APIs & Services → Library → Gmail API → Enable\n"
            "  4. Create OAuth 2.0 credentials:\n"
            "       APIs & Services → Credentials → Create Credentials\n"
            "       → OAuth client ID → Desktop app\n"
            "  5. Download the JSON file and save it as 'credentials.json'\n"
            "     in this project's root directory.\n"
            "  6. Re-run this script.\n"
        )
        sys.exit(1)

    print("Opening browser for Gmail OAuth2 authorisation…")
    from app.gmail_service import get_gmail_service
    service = get_gmail_service()

    # Quick connectivity test
    profile = service.users().getProfile(userId="me").execute()
    print(f"\n✅  Successfully authenticated as: {profile.get('emailAddress')}")
    print(f"    Token saved to: {settings.gmail_token_path}")
    print("\nYou can now start the application with:  python main.py\n")


if __name__ == "__main__":
    main()
