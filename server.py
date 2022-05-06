from gevent.pywsgi import WSGIServer

from web_app import app

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("运行http服务")
    parser.add_argument('--host', type=str, help='监听地址', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, help='监听端口', default=45883)
    args = parser.parse_args()

    WSGIServer((args.host, args.port), app).serve_forever()
