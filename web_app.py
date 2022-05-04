from flask import Flask, request, render_template

from cvt import im_read, image2txt

app = Flask(__name__)


@app.route('/cvt', methods=["POST"])
def cvt():
    image = im_read(request.files['image'])
    arg = request.form.to_dict()  # type:dict
    arg['image'] = image
    arg['col_x'] = int(arg['col_x'])
    arg['threshold'] = float(arg['threshold'])
    arg['no_black'] = 'no_black' in arg
    arg['image_inverse'] = 'image_inverse' in arg
    return image2txt(**arg)


@app.route('/')
def home():
    return render_template("home.html")


