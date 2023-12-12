from flask import Flask, jsonify, request

app = Flask(__name__)

received_data = {}  # Store received data globally

@app.route('/receive_text', methods=['POST', 'GET'])
def receive_text():
    global received_data

    if request.method == 'POST':
        data = request.json  # Get JSON data from the POST request
        received_data = data  # Store received data globally
        received_text = data.get('text')  # Extract 'text' from the JSON data
        print("Received text:", received_text)
        return "Text received by the server"

    elif request.method == 'GET':
        # Return the received data for the GET request
        #return jsonify(received_data), 200  # Returning received data with a status code
        return received_data.get('text'), 200

if __name__ == '__main__':
    app.run(debug=True)
