import argparse

import uvicorn

from .api import create_app
from .config import Settings
from .logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Run AgentMemory Token-authenticated memory APIs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    configure_logging()
    app = create_app(settings=Settings.from_toml(args.config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
