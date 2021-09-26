import multiprocessing as mp
import mp_tester as sim


def run():
    s = sim.Simulator()
    s.setup_simulator()
    s.run(n=3)


if __name__ == '__main__':
    mp.freeze_support()
    run()