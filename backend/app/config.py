from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CodeSentinel-AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # Origins allowed to call the API from a browser.
    #
    # 5500 is the VS Code Live Server default and is what the static frontend
    # in `frontend/` is served from during development. 3000 and 5173 cover
    # Create-React-App and Vite if the frontend is ever rebuilt on a bundler.
    # Both `localhost` and `127.0.0.1` are listed because browsers treat them
    # as different origins, and which one appears depends on how the developer
    # opened the page.
    #
    # Override in .env with a comma-separated list, e.g.
    #   CORS_ALLOW_ORIGINS=http://localhost:5500,https://example.com
    cors_allow_origins: str = (
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # Hard ceiling for uploaded files, enforced before the body is read into
    # the analyzer. The request schema caps `code` at 1 MB; this keeps the file
    # endpoint consistent with it.
    max_upload_bytes: int = 1_000_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_allow_origins` split into the list CORSMiddleware expects."""

        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
