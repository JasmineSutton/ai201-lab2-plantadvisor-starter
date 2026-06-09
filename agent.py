import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone.\n\n"
    "When lookup_plant returns found: False, do NOT invent specific care instructions for that plant. "
    "Instead, acknowledge clearly that the plant is not in your database, and offer general guidance "
    "based on what the user has described (e.g., if it sounds like a succulent, tropical, or fern, "
    "offer general advice for that category). You may suggest the user consult a resource like "
    "the American Horticultural Society for detailed species-specific data.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
)

# ──────────────────────────────────────────────
# Tool dispatch
# ──────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def run_agent(user_message: str, history: list) -> str:
    """
    Run the plant care agent for one user turn and return its response.

    The loop:
      1. Build messages list: system prompt + history + new user message
      2. Call the LLM with tool definitions
      3. If the response has tool_calls: execute them, append results, call LLM again
      4. Repeat until no tool_calls or MAX_TOOL_ROUNDS is reached
      5. Return the final text content
    """

    # Step 1: Build the messages list
    # Start with the system prompt so the LLM knows its role
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Replay conversation history so the LLM has context from prior turns
    # Gradio history is a list of [user_msg, assistant_msg] pairs
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # Step 2: Loop — call the LLM, handle tool calls, repeat
    for round_num in range(MAX_TOOL_ROUNDS):

        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except Exception as exc:
            # Groq can occasionally return a transient tool-use formatting error.
            # Retry once with an explicit protocol reminder instead of failing the turn.
            error_text = str(exc)
            if "tool_use_failed" in error_text:
                print(
                    f"  ⚠ Tool formatting error during round {round_num + 1}; retrying once"
                )
                retry_messages = messages + [{
                    "role": "system",
                    "content": (
                        "When using tools, return function calls with valid JSON arguments "
                        "that match the declared schema exactly."
                    ),
                }]
                try:
                    response = _client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=retry_messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                    )
                except Exception as retry_exc:
                    print(
                        f"  ✖ Retry failed during tool round {round_num + 1}: {retry_exc}"
                    )
                    return (
                        "I ran into a temporary issue while using my plant tools. "
                        "Please try again in a moment."
                    )
            else:
                print(f"  ✖ LLM call failed during tool round {round_num + 1}: {exc}")
                return (
                    "I ran into a temporary issue while using my plant tools. "
                    "Please try again in a moment."
                )

        assistant_message = response.choices[0].message

        # Step 3: Check if the LLM wants to call tools
        if not assistant_message.tool_calls:
            # No tool calls — the LLM has a final answer, exit the loop
            return assistant_message.content or "I'm not sure how to help with that."

        # Step 4: There are tool calls — append the assistant message FIRST
        # (the API requires the assistant message to appear before its tool results)
        messages.append(assistant_message)

        # Step 5: Execute each tool call and append results
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            tool_result = dispatch_tool(tool_name, tool_args)

            # Append the tool result — tool_call_id links it back to the request
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # Loop continues — LLM will now see the tool results and decide what to do next

    # Step 6: MAX_TOOL_ROUNDS reached — make one final call without tools to get a response
    print(f"  ⚠ MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}) reached — forcing final response")
    try:
        final_response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        return final_response.choices[0].message.content or "I reached my tool limit and couldn't complete the response."
    except Exception as exc:
        print(f"  ✖ Final fallback call failed: {exc}")
        return (
            "I reached my tool-call limit and couldn't finish the response this time. "
            "Please try rephrasing your question."
        )
