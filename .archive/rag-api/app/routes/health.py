from flask import Blueprint


health = Blueprint('health', __name__)


@health.route('/health', methods=['GET'])
def health():
    """
    Endpoint to check the health of the API.
    
    Returns:
        Tuple[str, int]: A tuple containing an empty string and the status code 204.
    """
    return '', 204
