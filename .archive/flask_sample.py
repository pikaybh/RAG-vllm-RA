from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
items = [
    {"id": 1, "name": "Item 1", "price": 100},
    {"id": 2, "name": "Item 2", "price": 200},
]

# Route to get all items
@app.route('/items', methods=['GET'])
def get_items():
    return jsonify({"items": items})

# Route to get a specific item by ID
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)

# Route to create a new item
@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    if not data or not all(key in data for key in ("name", "price")):
        return jsonify({"error": "Invalid data"}), 400

    new_item = {
        "id": max(item["id"] for item in items) + 1,
        "name": data["name"],
        "price": data["price"]
    }
    items.append(new_item)
    return jsonify(new_item), 201

# Route to update an item
@app.route('/items/<int:item_id>', methods=['PUT'])
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
@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [item for item in items if item["id"] != item_id]
    return jsonify({"message": "Item deleted"})

if __name__ == '__main__':
    app.run(debug=True)
