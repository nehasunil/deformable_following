from control.gripper.gripper_control import Gripper_Controller
from control.ur5.ur_controller import UR_Controller
import time
import numpy as np

urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()

# pose0 = np.array([-0.252, -0.138, 0.067, 0.013, -2.121, 2.313])
pose0 = np.array([-0.539, 0.312, 0.321, -1.787, -1.604, -0.691])
pose1 = np.array([-0.481, 0.283, 0.359, -1.6, -1.480, -1.031])
pose2 = np.array([-0.481, 0.283, 0.359, -1.216, -1.480, -.8])


# grc.gripper_helper.set_gripper_current_limit(0.4)


grc.follow_gripper_pos = 0
#
#
# a = 0.05
# v = 0.05
# urc.movel_wait(pose0, a=a, v=v)
# #
# c = input()

# urc.movel_wait(pose1, a=a, v=v)
# c = input()
# urc.movel_wait(pose0, a=a, v=v)
# c = input()
# urc.movel_wait(pose2, a=a, v=v)
#
# grc.follow_gripper_pos = 1
#
# c = input()

# time.sleep(0.5)
# # # pose1 = pose0 - [0.2, 0, 0, 0, 0, 0]
# # # #
# a = 0.02
# v = 0.02
# dt = 0.05
# # urc.movel_wait(pose1, a=a, v=v)
# for t in range(int(5//dt)):
#     urc.speedl( [-0.015, 0, 0, 0, 0, 0], a=a, t=dt)
#     grc.follow_gripper_pos -= .0005
#     time.sleep(dt)

print(', '.join([str("{:.3f}".format(_)) for _ in urc.getl_rt()]))
time.sleep(0.01)

urc.flag_terminate = True
# grc.flag_terminate = True
urc.join()
# grc.join()
