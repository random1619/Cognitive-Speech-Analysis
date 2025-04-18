import numpy as np
import math
from fastapi.responses import JSONResponse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_json_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization with better handling of special values."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN, Infinity, and -Infinity
        if np.isnan(obj):
            return None
        elif np.isposinf(obj):
            return "Infinity"  # Use string representation instead of large number
        elif np.isneginf(obj):
            return "-Infinity"  # Use string representation instead of large number
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    # Check for Python's built-in float special values
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj) and obj > 0:
            return "Infinity"
        elif math.isinf(obj) and obj < 0:
            return "-Infinity"
    return obj

class EnhancedJSONResponse(JSONResponse):
    """Custom JSON response that handles NaN and Infinity values with custom serialization."""
    
    def render(self, content):
        """Override render to use custom handling for special floats."""
        try:
            # First, make everything JSON serializable
            clean_content = make_json_serializable(content)
            
            # Handle special string values after serialization
            def replace_special_values(obj):
                if isinstance(obj, str) and obj in ["Infinity", "-Infinity"]:
                    return obj
                elif isinstance(obj, dict):
                    return {k: replace_special_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_special_values(item) for item in obj]
                return obj
            
            # Handle special values
            clean_content = replace_special_values(clean_content)
            
            # Then encode as JSON
            return json.dumps(
                clean_content,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")
        except Exception as e:
            logger.error(f"Error rendering JSON response: {str(e)}")
            # Fallback to basic error response
            return json.dumps(
                {"error": "Error rendering response"},
                ensure_ascii=False,
            ).encode("utf-8")