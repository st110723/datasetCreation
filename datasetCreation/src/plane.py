import os

import pybullet_data
import yaml


class Plane:

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        self.loadConfigFile()

    def loadConfigFile(self):
        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.debug = cfg.get("debugConfig").get("debug")

        self.useFile = cfg.get("planeConfig").get("useFile")
        if self.useFile:
            self.pathToUrdf = cfg.get("planeConfig").get("pathToUrdf")
            if self.debug:
                if not (os.path.exists(self.pathToUrdf)):
                    raise FileNotFoundError
        self.lateralFriction = cfg.get("planeConfig").get("lateralFriction")

    def load(self):
        if self.useFile:
            self.ID = self.bulletClient.loadURDF(self.pathToUrdf)
        else:
            self.bulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.ID = self.bulletClient.loadURDF("plane.urdf")
        self.set_friction()

    def set_friction(self):
        self.bulletClient.changeDynamics(self.ID, -1, lateralFriction=self.lateralFriction)

    def set_texture(self):
        textureId = self.bulletClient.loadTexture("meshes/plane/mars.png")
        self.bulletClient.changeVisualShape(self.ID, -1, textureUniqueId=textureId)
