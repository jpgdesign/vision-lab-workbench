"""Native ComfyUI nodes for the local Vision Workbench installation."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    from server import PromptServer
except ImportError:
    PromptServer = None
if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    from .routes import register_routes
    register_routes(PromptServer.instance)
