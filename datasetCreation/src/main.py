import os
from multiprocessing import Process

import yaml

from simulation import Simulation


def data_generation():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    workingDir = cfg.get("dirConfig").get("workingDir")
    os.chdir(workingDir)

    sim = Simulation()
    sim.init()
    sim.load_plane()
    sim.load_object()
    sim.load_rocks()
    sim.get_camera_image()
    sim.load_gripper()
    sim.step()


if __name__ == "__main__":
    numCpus = os.cpu_count() // 2
    # with Pool(numCpu) as p:
    #     p.map(data_generation, [1, 2])
    processes = []
    for _ in range(numCpus):
        proc = Process(target=data_generation)
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
