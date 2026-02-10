import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class GizmoConfig:
    # Database (PostgreSQL / Supabase)
    host: Optional[str] = None
    port: int = 6543
    user: Optional[str] = None
    password: Optional[str] = None
    database: str = "postgres"

    # Backblaze B2
    b2_endpoint: Optional[str] = None
    b2_key_id: Optional[str] = None
    b2_application_key: Optional[str] = None
    b2_bucket_name: str = "ChessGizmo"
    b2_region: str = "eu-central-003"

    # Lichess API
    lichess_token: Optional[str] = None

    @classmethod
    def from_env(cls):
        """
        Creates an instance of the config, automatically pulling values from .env or the environment.
        """
        load_dotenv()

        # Getting the port carefully so as not to fall into int(None)
        env_port = os.getenv('PORT')

        return cls(
            host=os.getenv('HOST'),
            port=int(env_port) if env_port else 6543,
            user=os.getenv('USER'),
            password=os.getenv('PASSWORD'),
            database=os.getenv('DATABASE', 'postgres'),

            b2_endpoint=os.getenv('B2_ENDPOINT'),
            b2_key_id=os.getenv('B2_KEY_ID'),
            b2_application_key=os.getenv('B2_APPLICATION_KEY'),
            b2_bucket_name=os.getenv('B2_BUCKET_NAME', 'ChessGizmo'),
            b2_region=os.getenv('B2_REGION', 'eu-central-003'),

            lichess_token=os.getenv('LICHESS_TOKEN')
        )

    def validate_db(self):
        """Checking database parameters"""
        required = ['host', 'user', 'password']
        self._check_required(required, "Database")

    def validate_b2(self):
        """Checking B2 storage parameters"""
        required = ['b2_endpoint', 'b2_key_id', 'b2_application_key']
        self._check_required(required, "Backblaze B2")

    def validate_lichess(self):
        """Lichess token verification"""
        if not self.lichess_token:
            raise ValueError("LICHESS_TOKEN must be provided for data fetching.")

    def _check_required(self, fields_list: list, service_name: str):
        missing = [f for f in fields_list if getattr(self, f) is None]
        if missing:
            raise ValueError(f"Missing required {service_name} fields: {', '.join(missing)}")

    def __post_init__(self):
        """Automatic type conversion if strings from env are passed."""
        if isinstance(self.port, str):
            self.port = int(self.port)