"""Local-only bounded job API for the production viewer."""

from .app import create_app

__all__ = ["create_app"]
