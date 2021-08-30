import urllib.request
import urllib
import cv2
import numpy as np

class Streaming(object):
    def __init__(self, stream):
        self.image = ""
        # self.url = url
        self.stream = stream

    def load_stream(self):
        stream = self.stream
        bytess=b''

        while True:
            bytess+=stream.read(32767)

            a = bytess.find(b'\xff\xd8') # JPEG start
            b = bytess.find(b'\xff\xd9') # JPEG end

            if a!=-1 and b!=-1:
                jpg = bytess[a:b+2] # actual image
                bytess= bytess[b+2:] # other informations

                self.image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR) 
