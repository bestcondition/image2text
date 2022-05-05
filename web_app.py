from flask import Flask, request, render_template

from text_image import im_read, image2text

app = Flask(__name__)


@app.route('/cvt', methods=["POST"])
def cvt():
    image = im_read(request.files['image'])
    arg = request.form.to_dict()  # type:dict
    return image2text(
        image=image,
        n=int(arg['n']),
        threshold=int(arg['threshold']),
        image_inverse='image_inverse' in arg
    )


@app.route('/')
def home():
    return render_template("home.html")
