def validate_prompt(prompt: str) -> tuple[bool, str]:
    if not prompt or not prompt.strip():
        return False, "Please enter a prompt."

    return True, ""
