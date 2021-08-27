import cv2
import numpy as np
import urllib

url = 'http://rpigelsight.local:8080/?action=stream'

try:
    # For python3
    import urllib.request
    stream = urllib.request.urlopen(url)

except:
    # For python2
    stream = urllib.urlopen(url)

bytess = b''
cnt = 0

while True:
    bytess += stream.read(32767)

    a = bytess.find(b'\xff\xd8') # JPEG start
    b = bytess.find(b'\xff\xd9') # JPEG end

    if a!=-1 and b!=-1:

        jpg = bytess[a:b+2] # actual image
        bytess = bytess[b+2:] # other informations

        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)

        cv2.imshow('frame', img)

        c = cv2.waitKey(1) 
        if c & 0xFF == ord('q'):
            break
