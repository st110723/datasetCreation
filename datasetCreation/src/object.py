import os

import numpy as np
import yaml

from basicObject import BasicObject
from mesh import Mesh
from urdf import URDF


class Object(BasicObject):

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        super().__init__(bulletClient=self.bulletClient)
        self.loadConfigFile()

        if self.generateUrdf:
            self.mesh = Mesh(pathToMesh=self.pathToMesh, unit=self.unit, density=self.density)
            self.urdf = URDF()
            self.name = os.path.splitext(os.path.split(self.pathToUrdf)[-1])[0]
            super().generate_urdf(scale=True)

    def loadConfigFile(self):

        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.debug = cfg.get("debugConfig").get("debug")
        self.generateUrdf = cfg.get("objectConfig").get("generateUrdf")
        if self.generateUrdf:
            self.pathToMesh = cfg.get("objectConfig").get("pathToMesh")
            if self.debug:
                if not (os.path.exists(self.pathToMesh)):
                    raise FileNotFoundError
            self.density = cfg.get("objectConfig").get("density")
            self.unit = cfg.get("objectConfig").get("unit")
        self.pathToUrdf = cfg.get("objectConfig").get("pathToUrdf")
        if (not self.generateUrdf) and (self.debug):
            if not (os.path.exists(self.pathToUrdf)):
                raise FileNotFoundError
        self.mode = cfg.get("objectConfig").get("mode")
        if self.debug:
            if self.mode not in ["random", "fixed"]:
                raise ValueError("Invalid mode")
        if self.mode == "fixed":
            self.startPosition = cfg.get("objectConfig").get("startPosition")
            self.startOrientation = cfg.get("objectConfig").get("startOrientation")
        self.lateralFriction = cfg.get("objectConfig").get("lateralFriction")

    def load(self):
        if self.mode == "random":
            self.startPosition = [0, 0, 1]
            self.startOrientation = [0, 0, np.asscalar(np.random.default_rng().uniform(-np.pi, np.pi, 1))]

        super().load()
        super().set_friction()
        super().set_texture(color=[0.4, 0.2, 0, 1])
