import os


def init2(default_mcp=None):
    """Initialize MCP (market clearing price).

    Behavior:
    - If `default_mcp` is provided (numeric), use it.
    - Else if environment variable `MCP` is set, use that.
    - Otherwise fallback to a safe default (7.0) and print a notice.
    This avoids interactive prompts during automated runs.
    """
    global MCP

    if default_mcp is not None:
        MCP = float(default_mcp)
        print(f"MCP set from argument: {MCP}")
        return

    env_val = os.environ.get('MCP')
    if env_val is not None:
        try:
            MCP = float(env_val)
            print(f"MCP set from environment: {MCP}")
            return
        except Exception:
            pass

    # safe default to allow automated runs
    MCP = 7.0
    print(f"MCP not provided; using default MCP={MCP}")
    