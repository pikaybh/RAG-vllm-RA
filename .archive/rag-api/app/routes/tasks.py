# from typing import Dict, List
from flask import Blueprint, jsonify, request

from app.models import create_rag_chain


tasks = Blueprint('tasks', __name__)

# Tasks handler
# tasks: List[dict] = []


@tasks.route('/tasks', methods=['POST'])
def create_task(model: str):
    """
    Endpoint to create a new task.
    
    Args:
        model (str): The name of the model to use.
    Returns:
        Tuple[Dict[str, List[Dict[str, str]]], int]: A tuple containing the list of tasks and the status code 201.
    """
    rag_chain = create_rag_chain(model)
    data: dict = request.get_json()
    if not data or not all(key in data for key in ("messages")):
        return jsonify({"error": "Invalid data"}), 400

    new_task: dict = {
        # "id": max(task["id"] for task in tasks) + 1 if tasks else 1,
        "messages": data["messages"]
    }
    rst: str = rag_chain.run(new_task["messages"]["content"])
    # tasks.append(new_task)
    _payload = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": rst
                }
            }
        ]
    }
    return jsonify(_payload), 201

"""
# Route to update an item
@v1.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.get_json()
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404

    item.update({
        "name": data.get("name", item["name"]),
        "price": data.get("price", item["price"])
    })
    return jsonify(item)

# Route to delete an item
@v1.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [item for item in items if item["id"] != item_id]
    return jsonify({"message": "Item deleted"})
"""