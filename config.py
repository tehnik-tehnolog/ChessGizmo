import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class GizmoConfig:
    load_dotenv()
    # Database (PostgreSQL / Supabase)
    host: str = os.getenv('HOST')
    port: int = int(os.getenv('PORT', 6543)) # for poller
    user: str = os.getenv('USER')
    password: str = os.getenv('PASSWORD')

    # Backblaze B2
    b2_endpoint: str = os.getenv('B2_ENDPOINT')
    b2_key_id: str = os.getenv('B2_KEY_ID')
    b2_application_key: str = os.getenv('B2_APPLICATION_KEY')
    b2_bucket_name: str = os.getenv('B2_BUCKET_NAME', 'ChessGizmo')
    b2_region: str = os.getenv('B2_REGION', 'eu-central-003')

    # Lichess API
    lichess_token: str = os.getenv('LICHESS_TOKEN')