from typing import Optional
from nemoguardrails.actions import action


@action(is_system_action=True)
async def self_check_input(context: dict):
    """
    Check if user input should be blocked based on content guidelines.
    Returns True if input should be blocked, False if allowed.
    """
    user_message = context.get("user_message", "").lower()

    # Define blocked keywords/patterns
    blocked_keywords = [
        "hack", "hacking", "exploit", "crack",
        "order", "track", "return", "refund", "shipping", "delivery",
        "impersonate", "pretend to be", "act as",
        "forget", "ignore", "override", "bypass"
    ]

    # Check for blocked content
    for keyword in blocked_keywords:
        if keyword in user_message:
            print(f"Blocked keyword found: {keyword}")
            return True

    # Allow smartphone-related queries
    smartphone_keywords = [
        "iphone", "samsung", "galaxy", "phone", "smartphone",
        "specs", "features", "comparison", "camera", "battery",
        "processor", "ram", "storage", "display", "price", "android", "ios"
    ]

    # If it contains smartphone keywords, likely safe
    if any(keyword in user_message for keyword in smartphone_keywords):
        print(f"Smartphone-related query detected, allowing")
        return False

    # For general questions, be permissive
    print(f"General query, allowing")
    return False