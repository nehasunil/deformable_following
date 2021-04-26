import threading
import time
from scipy.spatial.transform import Rotation as R
import copy
import open3d as o3d
from open3d import *
import numpy as np

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, target):
        threading.Thread.__init__(self)
        self.target = copy.deepcopy(target)
        self.stop = False
        self.id = -1
        self.reg_p2p = None
        self.running = False
        
    def icp(self):
        threshold = 2
        trans_init = self.trans_init.copy()
        
        rot_noise = np.eye(4)
        rot_noise[:3,:3] = R.from_euler('xyz', np.random.random(3)*3-1.5, degrees=True).as_matrix()
        
        trans_init =  rot_noise.dot(trans_init)
        target = copy.deepcopy(self.target)
        # trans_init[:,:3] += np.random.random(3) * 0.01
        
        reg_p2p = o3d.registration.registration_icp(
            self.source, target, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=1))
        self.reg_p2p = reg_p2p
        
    def run(self):
        last_id = -1
        while True:
            if self.stop:
                break
            time.sleep(0.001)
            current_id = self.id
            if current_id > last_id:
                self.running = True
                self.icp()
                self.running = False
                last_id = current_id
                
                

class ThreadICP():
    def __init__(self, target=None, nb_worker=4, time_limit=0.1):
        self.nb_worker = nb_worker
        self.time_limit = time_limit
        self.thread_list = []

        # Create new threads
        for i in range(self.nb_worker):
            t = myThread(target)
            self.thread_list.append(t)

        # Start new Threads
        for t in self.thread_list:
            t.start()

        self.frame_id = 0
    
    def estimate(self, source, trans_init):
        for t in self.thread_list:
            if t.running == False:
                t.source = source
                t.trans_init = trans_init
                t.id = self.frame_id
            

        time.sleep(self.time_limit)
        
        trans_opt = trans_init.copy()
        fitness_max = -1
        inlier_rmse_min = 1e9
        
        
        for t in self.thread_list:
            st = time.time()
            if t.id == self.frame_id and t.reg_p2p is not None:
                print(f"fitness {t.reg_p2p.fitness} inlier_rmse {t.reg_p2p.inlier_rmse}")
        
                if t.reg_p2p.fitness > fitness_max or \
                (t.reg_p2p.fitness == fitness_max and t.reg_p2p.inlier_rmse < inlier_rmse_min):
                    fitness_max = t.reg_p2p.fitness
                    inlier_rmse_min = t.reg_p2p.inlier_rmse
                    trans_opt = t.reg_p2p.transformation.copy()
            print("time", time.time()-st)
        
        self.frame_id += 1
        print(f"fitness_max {fitness_max} inlier_rmse_min {inlier_rmse_min}")
        
        return trans_opt

    def stop(self):
        for t in self.thread_list:
            t.stop = True
            t.join()

        print ("Exiting Main Thread")
    

if __name__ == '__main__':
    thread_icp = ThreadICP(target, nb_worker=3, time_limit=0.1)

    for i in range(4):
        trans_opt = thread_icp.estimate(source, trans_init)
        print(trans_opt)

    thread_icp.stop()
