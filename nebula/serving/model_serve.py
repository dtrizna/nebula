from http.server import BaseHTTPRequestHandler, HTTPServer
from base64 import b64decode
import logging
import argparse
import pickle
import time

from xgboost import XGBClassifier

import sys
sys.path.extend(["..", "."])
from nebula.misc import getRealPath
SCRIPT_PATH = getRealPath(type="script")


class MyHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self,
        model = None,
        encoder = None,
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder

    def _set_response_headers(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self.send_error(405, "Use POST body with base64 encoded command to perform inference!")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()

        logging.info(f"""Path: {str(self.path)}
Headers:\n{str(self.headers)}

Body:\n{post_data}\n""")
        
        xgb_model = XGBClassifier(n_estimators=100, 
                                  use_label_encoder=False, 
                                  eval_metric="logloss")
                
        if self.path == "/":
            encoder_path = self.encoder
            model_path = self.model
        else:
            encoder_path = None
            model_path = None
            self._set_response_headers(code=404)

        if encoder_path and model_path:
            encoder = pickle.load(open(encoder_path, "rb"))            
            xgb_model.load_model(model_path)
            
            try:
                cmd = b64decode(post_data).decode()
            except:
                cmd = str(post_data)

            #logging.info(f"Received command to verify: {cmd}")
            x = encoder.transform([cmd])
            prediction = xgb_model.predict_proba(x)
            msg = f"Input:{cmd}\nProb(malicious): {prediction[0][1]*100:.6f}%\n"
            logging.info(msg)
            self._set_response_headers()
            self.wfile.write(msg.encode())


def run(server_class=HTTPServer, 
        handler_class=MyHTTPHandler, 
        model_path = None,
        port=8080, 
        address="0.0.0.0"):
    server_address = (address, port)
    httpd = server_class(server_address, handler_class)
    handler_class.path = model_path
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    
    # model parameters
    parser.add_argument("--model-path", type=str, default="TfidfVectorizer_max500_ngram12", help="Path to model file.")
    
    # server parameters
    parser.add_argument("--port", type=int, default=80, help="Port to serve the model via HTTP.")
    parser.add_argument("--address", type=str, default="0.0.0.0", help="Address to bind HTTP server.")
    # auxiliary
    parser.add_argument("--logfile", type=str, help="File to store logging messages.")
    parser.add_argument("-d", "--debug", action="store_true", help="Provide with DEBUG level information from packages.")

    args = parser.parse_args()
    
    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    logging.warning(f" [*] {time.ctime()}: Staring server on {args.address}:{args.port}")
    run(port=args.port, address=args.address, model_path=f"{SCRIPT_PATH}/../models/{args.model_path}")
    logging.warning(f" [*] {time.ctime()}: Stopped server on {args.address}:{args.port}")