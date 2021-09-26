import os
import multiprocessing as mp
import pybullet as p
from pybullet_utils import bullet_client as bc


class Simulator():

    hz = 120
    gravity = (0, 0, -9.8)

    def __init__(self):
        self.p_client = None # Physics client for this instance
        self.data_list = range(100)

    def setup_simulator(self, main=False, freq=hz):
        self.p_client = bc.BulletClient(connection_mode=p.DIRECT)
        print(f'Process client: {self.p_client._client}')
        self.p_client.setGravity(*self.gravity)
        self.p_client.setPhysicsEngineParameter(fixedTimeStep=1.0 / freq)

    def run(self, n=1):
        jobs = []
        results = mp.Queue()
        for i in range(n):
            print(f'Starting Job:{i}')
            proc = mp.Process(target=self.init_mp_sim, args=(i, results,))
            jobs.append(proc)
            proc.start()

        print('Getting results')
        res = [results.get() for i in range(n)]
        for pr in jobs:
            pr.join()
        print(f'Returned data: {res}')

    def init_mp_sim(self, i=None, q=None):
        self.setup_simulator()
        self.p_client.stepSimulation()
        print(f'Loaded and set gravity for process {i}')
        q.put([i])
