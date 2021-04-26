import urx
import time
import socket
import numpy as np

HOST = "10.42.0.121"
PORT = 30003

rob = urx.Robot(HOST, use_rt=True)
time.sleep(0.5)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))


pose = rob.rtmon.getTCF(True)
print(pose)

pose0 = np.array([-0.45787275, 0.21882309, 0.42330343, -1.21460518, -1.23206398, -1.04391101])
pose1 = pose0 - [0, 0, 0.03, 0, 0, 0]
a = 0.1
v = 0.2
cmd = f"movel(p{list(pose1)}, a={a}, v={v})" + "\n"
cmd = str.encode(cmd)
s.send(cmd)

time.sleep(1)

cmd = f"movel(p{list(pose0)}, a={a}, v={v})" + "\n"
cmd = str.encode(cmd)
s.send(cmd)

print(pose)

rob.close()