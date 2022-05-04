from gevent.pywsgi import WSGIServer

from web_app import app

if __name__ == '__main__':
    WSGIServer(('127.0.0.1', 45883), app).serve_forever()
