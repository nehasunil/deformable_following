import serial
from serial.tools import list_ports

class AutoSerial:

    def __init__(self, vendor_id, product_id, serial_number, serial_port=None, baud_rate=None, connect=True):
        self.ser = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.serial_number = serial_number
        self.target_string = "USB VID:PID=%s:%s"%(self.vendor_id, self.product_id)
        self.detected_port = None

        if self.ser is None:
            self.detected_port = None
            print("target: ", self.target_string)
            for port in list(list_ports.comports()):
                print(port[2], self.target_string in port[2])

                if self.target_string in port[2]:
                    self.detected_port = port[0]

            if self.detected_port is None:
                print("Automatic serial port connection failed, using custom serial port...")
                print("custom port: ", self.serial_port, self.baud_rate)
                if connect:
                	self.ser = serial.Serial(self.serial_port, self.baud_rate)
            else:
                print("detected port: ", self.detected_port, self.baud_rate)
                if connect:
                	self.ser = serial.Serial(self.detected_port, self.baud_rate)
