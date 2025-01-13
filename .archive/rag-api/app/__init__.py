from flask import Flask
from .routes import blueprints


__version__ = "1.0.0"


def start_server(host: str = "0.0.0.0", port: int = 8000, *args):
    app = Flask(__name__)

    # Register Blueprint
    for blueprint in blueprints:
        app.register_blueprint(blueprint, url_prefix=f'/v{__version__.split(".")[0]}')

    app.run(host=host, port=port, 
            debug = True if "debug" in args else False)
