import socket
import time

HOST = "10.42.1.121"    # The remote host
PORT = 30002              # The same port as used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
time.sleep(0.5)
s.send("movel(p[-0.5, -0., 0.35, 0.1, 4.76, 0.09], a=0.01, v=0.01)" + "\n")

s.close()