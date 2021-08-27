from struct import *
import time
from .auto_serial import AutoSerial


class SBMotor:
    def __init__(self, serial_port, baud_rate=115200):
        self.serial_port = serial_port
        self.motor_ids = []
        self.baud_rate = baud_rate

        self.motor_reg_space = 16
        self.counter = 0
        self.time = time.time()

        auto_serial = AutoSerial(
            serial_port=self.serial_port,
            baud_rate=self.baud_rate,
            vendor_id="16C0",
            product_id="0483",
            serial_number="8218580",
        )

        self.ser = auto_serial.ser

        self.recv_buffer = bytearray(62)  # with current
        # self.recv_buffer = bytearray(58)  # without current
        self.enable = [False] * 8
        self.stop = [False] * 8
        self.pos = [0.0] * 8
        self.vel = [0.0] * 8
        self.curr = [0.0] * 8

    #     if motor_params is None:
    #         (self.ticks_per_rev, self.kp, self.ki, self.kd) = self.get_default_params()
    #     else:
    #         (self.ticks_per_rev, self.kp, self.ki, self.kd) = motor_params

    #     print(self.ticks_per_rev, self.kp, self.ki, self.kd)
    #     self.init(self.ticks_per_rev, self.kp, self.ki, self.kd)

    # def get_default_params(self):
    #     if self.motor_rpm == 26:
    #         params = (5462.22, 25.0, 0, 0.16)
    #     elif self.motor_rpm == 44:
    #         params = (3244.188, 15.0, 0, 0.2)
    #     elif self.motor_rpm == 52:
    #         params = (2774.64, 16, 0, 0.3)
    #     elif self.motor_rpm == 280:
    #         params = (514.14, 25.0, 0, 0.16)
    #     elif self.motor_rpm == 2737:
    #         params = (52.62, 100.0, 0, 0.16)
    #     elif self.motor_rpm == 130:
    #         params = (3040.7596, 14.0, 0, 0.3)
    #     else:
    #         params = ()
    #         print("motor not supported")
    #     return params

    def send_cmd_four_double(self, register_name, val1, val2, val3, val4):
        # ser = serial.Serial(self.serial_port, self.baud_rate)
        # print('send_cmd_four_double')
        data = bytearray(
            pack(
                "<BBBBHddddxx",
                0x7E,
                0,
                0xFF,
                0xAA,
                register_name,
                val1,
                val2,
                val3,
                val4,
            )
        )
        # print('unstuffed: ', data)
        data = self.msg_stuff(data)
        # print('stuffed: ', data)
        self.ser.write(data)

    def send_cmd_eight_float(self, register_name, data_eight_vals):
        data = bytearray(
            pack(
                "<BBBBHffffffffxx", 0x7E, 0, 0xFF, 0xAA, register_name, *data_eight_vals
            )
        )
        data = self.msg_stuff(data)
        self.ser.write(data)

    def send_cmd_twelve_float(self, register_name, data_twelve_vals):
        data = bytearray(
            pack(
                "<BBBBHffffffffffffxx",
                0x7E,
                0,
                0xFF,
                0xAA,
                register_name,
                *data_twelve_vals
            )
        )
        data = self.msg_stuff(data)
        self.ser.write(data)

    def send_cmd_single_double(self, register_name, val):
        # ser = serial.Serial(self.serial_port, self.baud_rate)

        data = bytearray(pack("<BBBBHdxx", 0x7E, 0, 0xFF, 0xAA, register_name, val))
        data = self.msg_stuff(data)
        self.ser.write(data)

    def send_cmd_single_int(self, register_name, val):
        # ser = serial.Serial(self.serial_port, self.baud_rate)
        # print('send_cmd_single_int')
        data = bytearray(pack("<BBBBHixx", 0x7E, 0, 0xFF, 0xAA, register_name, val))
        # print('unstuffed: ', data)
        data = self.msg_stuff(data)
        # print('stuffed: ', data)
        self.ser.write(data)

    def send_cmd_single_bool(self, register_name, val):
        # ser = serial.Serial(self.serial_port, self.baud_rate)
        data = bytearray(pack("<BBBBH?xx", 0x7E, 0, 0xFF, 0xAA, register_name, val))
        data = self.msg_stuff(data)
        self.ser.write(data)

    def init_single_motor(self, motor_id, ticks_per_rev, kp, ki, kd, ctrl_mode):
        if motor_id not in self.motor_ids:
            self.motor_ids.append(motor_id)
        # init pid
        register_name = motor_id * self.motor_reg_space + 16 + 9
        self.send_cmd_four_double(register_name, ticks_per_rev, kp, ki, kd)
        # set control mode, 0 pos, 1 vel
        register_name = motor_id * self.motor_reg_space + 16 + 10
        self.send_cmd_single_int(register_name, ctrl_mode)

    def set_enable(self, motor_id, if_enable):
        register_name = motor_id * self.motor_reg_space + 16
        self.enable[motor_id] = if_enable
        self.send_cmd_single_bool(register_name, self.enable[motor_id])

    def set_stop(self, motor_id, if_stop):
        register_name = motor_id * self.motor_reg_space + 16 + 7
        self.stop[motor_id] = if_stop
        self.send_cmd_single_bool(register_name, self.stop[motor_id])

    def move_to_pos(self, motor_id, degree):
        register_name = motor_id * self.motor_reg_space + 16 + 1
        self.pos[motor_id] = degree
        self.send_cmd_single_double(register_name, self.pos[motor_id])

    def set_speed(self, motor_id, rpm):
        register_name = motor_id * self.motor_reg_space + 16 + 2
        self.vel[motor_id] = rpm
        self.send_cmd_single_double(register_name, self.vel[motor_id])

    def set_current(self, motor_id, curr):
        register_name = motor_id * self.motor_reg_space + 16 + 10
        self.curr[motor_id] = curr
        self.send_cmd_single_int(register_name, self.curr[motor_id])

    def move(self, motor_id, degree):
        register_name = motor_id * self.motor_reg_space + 16 + 1
        self.pos[motor_id] += degree
        self.send_cmd_single_double(register_name, self.pos[motor_id])

    def set_kp(self, motor_id, kp):
        register_name = motor_id * self.motor_reg_space + 16 + 3
        self.kp = kp
        self.send_cmd_single_double(register_name, self.kp)

    def set_ki(self, motor_id, ki):
        register_name = motor_id * self.motor_reg_space + 16 + 4
        self.ki = ki
        self.send_cmd_single_double(register_name, self.ki)

    def set_kd(self, motor_id, kd):
        register_name = motor_id * self.motor_reg_space + 16 + 5
        self.kd = kd
        self.send_cmd_single_double(register_name, self.kd)

    def request_vals(self):
        register_name = 0xE0
        self.send_cmd_single_int(register_name, 53)

    def move_all_to_pos(self, positions):
        # The positions contain only the positions of the active motors
        for ii in range(len(self.motor_ids)):
            self.pos[self.motor_ids[ii]] = positions[ii]
        register_name = 0xD1
        self.send_cmd_eight_float(register_name, self.pos)

    def set_all_velocity(self, velocities):
        # The positions contain only the positions of the active motors
        for ii in range(len(self.motor_ids)):
            self.vel[self.motor_ids[ii]] = velocities[ii]
        register_name = 0xD2
        self.send_cmd_eight_float(register_name, self.vel)

    @staticmethod
    def msg_stuff(msg):
        msg_len = len(msg)
        msg[1] = msg_len - 2
        stuffing = 2
        for ii in range(1, msg_len):
            # print("%x"%msg[ii])
            if msg[ii] == 0x7E:
                # print(ii)
                msg[stuffing] = ii
                stuffing = ii
        msg[stuffing] = 0xFF
        return msg

    @staticmethod
    def msg_unstuff(msg):
        stuffing = 2
        while msg[stuffing] != 0xFF:
            tmp = msg[stuffing]

            msg[stuffing] = 0x7E
            stuffing = tmp
            # print(len(msg))
            # print(stuffing)
        msg[stuffing] = 0x7E
        return msg

    def recv_from_serial(self):
        last_state = -1
        cur_state = 0
        rx_len = 0
        idx = 0
        # ready_to_return = False
        return_val = None

        while self.ser.in_waiting:
            # rx = self.ser.read(1)

            if cur_state == 0:
                # if last_state != cur_state:
                #   print("read header")
                rx = self.ser.read(1)
                # print(rx)
                # print("%x" % rx[0])
                if int(rx[0]) == 0x7E:
                    cur_state = 1
                    idx = 0
                    self.recv_buffer[idx] = rx[0]
                last_state = 0

            elif cur_state == 1:
                # if last_state != cur_state:
                #   print("read length")
                rx = self.ser.read(1)
                rx_len = int(rx[0])
                # print("%x" % rx[0])
                if rx_len <= 0:
                    cur_state = 3
                else:
                    self.recv_buffer[1] = rx[0]
                    cur_state = 2
                last_state = 1

            elif cur_state == 2:
                # if last_state != cur_state:
                #   print("read data")
                rx = self.ser.read(1)
                # print("%x" % rx[0])
                if int(rx[0]) == 0x7E:
                    cur_state = 3
                else:
                    self.recv_buffer[idx + 2] = rx[0]
                    idx += 1
                    # print(idx)
                    if idx + 2 == rx_len:
                        self.recv_buffer = self.msg_unstuff(bytearray(self.recv_buffer))

                        # unpacked_data = unpack('<BBBBHlffffffffffff', self.recv_buffer)
                        # return_val = unpacked_data[6:]
                        unpacked_data = unpack(
                            "<BBBBHffffffffffffhhhh", self.recv_buffer
                        )
                        return_val = unpacked_data[5:]
                        # print(unpacked_data)

                        # cur_time = time.time()
                        # if (cur_time - self.time) > 0.05:
                        #   print(unpacked_data)
                        #   # motor0.counter = 0
                        #   # print(cur_time - self.time)
                        #   self.time = cur_time
                        cur_state = 0
                        self.counter += 1

                last_state = 2
            if cur_state == 3:
                # if last_state != cur_state:
                #   print("read error")
                last_state = cur_state
                cur_state = 0
        if return_val is not None:
            return [return_val[i] for i in range(8)]

        return None


if __name__ == "__main__":
    import numpy as np

    motor_cpr = 3040.7596
    com_baud = 1000000
    print("Establishing Serial port...")
    dc_motors = SBMotor("/dev/cu.usbmodem123", com_baud)
    print(dc_motors.ser)
    dc_motors.init_single_motor(2, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1)
    dc_motors.init_single_motor(3, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1)
    # dc_motors.init_single_motor(6, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0)
    # dc_motors.init_single_motor(7, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0)
    # dc_motors.init_single_motor(4, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1)
    dc_motors.init_single_motor(4, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0)
    # dc_motors.init_single_motor(5, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1)
    dc_motors.init_single_motor(5, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0)

    input("finish init")
    # pos_list = [[0, 0], [-40, 0], [0, 0], [0, 40], [0, 0], [-20, 20]]
    pos_list = [[0, 0], [-15, 15]]
    pos_cnt = 0
    pos = 0
    st_time = time.time()
    while True:
        curr_time = time.time()
        # vel = np.sin(curr_time) * 1
        vel = 10
        pos = np.sin((curr_time - st_time) * 4) * 20
        c = input()
        # pos = 10
        pos_cnt = (pos_cnt + 1) % len(pos_list)

        dc_motors.set_all_velocity([0, 0, pos_list[pos_cnt][0], pos_list[pos_cnt][1]])
        print(dc_motors.vel)
        dc_motors.request_vals()
        if dc_motors.ser.in_waiting:
            sensor_val = dc_motors.recv_from_serial()
            print("sensor_val: ", sensor_val)

        # time.sleep(0.1)
