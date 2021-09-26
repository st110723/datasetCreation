import os

import yaml

from basicObject import BasicObject
from mesh import Mesh
from urdf import URDF


class Rock(BasicObject):

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        super().__init__(self.bulletClient)
        self.loadConfigFile()

        if self.generateUrdf:
            self.mesh = Mesh(pathToMesh=self.pathToMesh, density=self.density, unit=self.unit)
            self.urdf = URDF()
            self.name = os.path.splitext(os.path.split(self.pathToUrdf)[-1])[0]
            super().generate_urdf(scale=False)

    def loadConfigFile(self):

        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

            self.debug = cfg.get("debugConfig").get("debug")
            self.generateUrdf = cfg.get("rockConfig").get("generateUrdf")
            if self.generateUrdf:
                self.pathToMesh = cfg.get("rockConfig").get("pathToMesh")
                if self.debug:
                    if not (os.path.exists(self.pathToMesh)):
                        raise FileNotFoundError
                self.density = cfg.get("rockConfig").get("density")
                self.unit = cfg.get("rockConfig").get("unit")
            self.pathToUrdf = cfg.get("rockConfig").get("pathToUrdf")
            if (not self.generateUrdf) and (self.debug):
                if not (os.path.exists(self.pathToUrdf)):
                    raise FileNotFoundError
            self.lateralFriction = cfg.get("rockConfig").get("lateralFriction")
            self.spinningFriction = cfg.get("rockConfig").get("spinningFriction")
            self.rollingFriction = cfg.get("rockConfig").get("rollingFriction")

    def load(self):
        super().load()
        super().set_friction()
        super().set_texture(color=[0.35, 0.3, 0.25, 1])
