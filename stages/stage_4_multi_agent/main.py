"""Compatibility wrapper for the corrected Stage 4 path.

The original codelab folder is named stage_4_milti_agent. README examples use
stage_4_multi_agent, so this wrapper keeps both commands working.
"""

from stages.stage_4_milti_agent.main import *  # noqa: F401,F403
from stages.stage_4_milti_agent.main import main


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
