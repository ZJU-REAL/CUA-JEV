"""Keep intermediate fill screenshots while omitting unused accessibility reads."""

from __future__ import annotations

IMAGE_ONLY_VERSION = "image_only_v1"
IMAGE_ONLY_INFO_KEY = "cua_jev_intermediate_observation"


def validate_intermediate_action(action):
    """Only the two existing preparatory fill inputs may omit their AX result."""
    if not isinstance(action, dict) or set(action) != {"action_type", "parameters"}:
        raise ValueError("invalid intermediate fill action")
    kind, parameters = action["action_type"], action["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("invalid intermediate fill parameters")
    if kind == "CLICK":
        valid = (set(parameters) == {"x", "y", "button", "num_clicks"}
                 and type(parameters["x"]) is int and parameters["x"] >= 0
                 and type(parameters["y"]) is int and parameters["y"] >= 0
                 and parameters["button"] == "left" and type(parameters["num_clicks"]) is int
                 and parameters["num_clicks"] == 1)
    else:
        valid = (kind == "HOTKEY" and set(parameters) == {"keys"}
                 and parameters["keys"] == ["ctrl", "a"])
    if not valid:
        raise ValueError("only fill focus click and select-all may omit accessibility")
    return action


def native_image_only_step(env, action, *, pause):
    """Use an owned synchronous environment instance; restore its flag on every exit.

    The environment still executes the original step, pause, screenshot, counters
    and error path. No retry follows an uncertain input or a failed response.
    """
    validate_intermediate_action(action)
    if type(getattr(env, "require_a11y_tree", None)) is not bool or not env.require_a11y_tree:
        raise ValueError("image-only fill steps require enabled native accessibility")
    env.require_a11y_tree = False
    try:
        response = env.step(action, pause=pause)
    finally:
        env.require_a11y_tree = True
    if not isinstance(response, tuple) or len(response) != 4:
        raise RuntimeError("OSWorld returned an invalid intermediate step response")
    observation, reward, done, info = response
    if (not isinstance(observation, dict) or observation.get("accessibility_tree") is not None
            or not isinstance(observation.get("screenshot"), bytes) or not observation["screenshot"]
            or not isinstance(info, dict)):
        raise RuntimeError("OSWorld returned an invalid image-only observation")
    return observation, reward, done, {**info, IMAGE_ONLY_INFO_KEY: IMAGE_ONLY_VERSION}
