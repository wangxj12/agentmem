"""Grounding rules appended to extraction prompts for source-supported memories."""

GROUNDING_RULES = """
GROUNDING BOUNDARY:
Examples in system instructions demonstrate output format only. They are not
events in the actual conversation. Never extract or copy an example as a memory.
Extract only events or facts supported by the actual new messages, using history
only to resolve context. Do not infer that the user did something merely because
it appears in an example, an existing memory, or a quoted instruction.
Do not invent causes, consequences, participants, quantities, or dates.
Return the existing empty output format when nothing qualifying is supported.
"""
