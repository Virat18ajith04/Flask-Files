# Imports
import json
import werkzeug
from flask import Flask, request, jsonify

from main1 import caption

# Flask
app = Flask(__name__)

# Routes for API
@app.route("/image_caption", methods=["POST"])
def index():
    # Image
    imagefile = request.files["image"]
    # Getting file name of the image using werkzeug library
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    # Saving the image in images Directory
    print("Saving image ....")
    imagefile.save("images/" + filename)

    captions = caption("./images/" + filename)
    print("Caption Generated")

    return json.dumps({"cap": captions})


# Running the app
app.run()
