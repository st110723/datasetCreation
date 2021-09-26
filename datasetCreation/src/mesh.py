import numpy as np
import pybullet as pb
import trimesh
import yaml


class Mesh:

    def __init__(self, pathToMesh, unit, density):

        self.pathToMesh = pathToMesh
        self.mesh = self.load_mesh(self.pathToMesh)

        assert unit in ["mm", "cm", "dm", "m"], "Invalid mesh unit"
        if unit == "mm":
            self.scale(0.001)
        elif unit == "cm":
            self.scale(0.01)
        elif unit == "dm":
            self.scale(0.1)
        else:
            pass

        self.density = density
        self.set_density(self.density)

    def load_mesh(self, pathToMesh):
        mesh = trimesh.load(pathToMesh)
        return mesh

    def get_bounding_box_dimensions(self):
        length, depth, height = np.max(self.boundingBox, axis=0) - np.min(self.boundingBox, axis=0)
        return length, depth, height

    def get_scale_factor(self):
        # gripperWidth = 0.085
        # shortestSide = np.min(np.array(self.get_bounding_box_dimensions()))
        # scaleFactor = 0.70 * gripperWidth / shortestSide

        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        fov = cfg.get("cameraConfig").get("fov")
        farVal = cfg.get("cameraConfig").get("farVal")
        longestSide = np.max(np.array(self.get_bounding_box_dimensions()))
        height = 2 * farVal * np.tan(np.deg2rad(fov / 2))
        scaleFactor = 0.8 * height / longestSide
        return scaleFactor

    def scale(self, scaleFactor):
        scaleMatrix = np.eye(4)
        scaleMatrix[:3, :3] = np.diag(np.repeat(scaleFactor, 3))
        self.mesh.apply_transform(scaleMatrix)

    def set_density(self, density):
        self.mesh.density = density

    @property
    def boundingBox(self):
        return self.mesh.bounds

    @property
    def centerOfMass(self):
        return self.mesh.center_mass

    @property
    def mass(self):
        return self.mesh.mass

    @property
    def volume(self):
        return self.mesh.volume

    @property
    def inertia(self):
        inertiaTensor = self.mesh.moment_inertia
        inertiaDict = {"ixx": inertiaTensor[0, 0],
                       "ixy": inertiaTensor[0, 1],
                       "ixz": inertiaTensor[0, 2],
                       "iyy": inertiaTensor[1, 1],
                       "iyz": inertiaTensor[1, 2],
                       "izz": inertiaTensor[2, 2]}
        return inertiaDict

    def change_origin(self, origin):
        self.mesh.vertices -= origin

    def decompose(self, pathToMesh, pathToNewMesh):
        # self.collisionMesh.visual = trimesh.visual.ColorVisuals()
        # self.collisionMesh.export(self.pathToCollisionMesh)

        # meshList = self.mesh.convex_decomposition(resolution=1000000,
        #                                           concavity=0.001,
        #                                           gamma=0.0005)
        # self.mesh = trimesh.util.concatenate(meshList)

        pb.vhacd(pathToMesh,
                 pathToNewMesh,
                 "logs/vhacd",
                 resolution=1000000,
                 concavity=0.001,
                 gamma=0.0005)

        newMesh = Mesh(pathToNewMesh, unit="m", density=self.density)
        return newMesh

    def save(self):
        self.mesh.visual = trimesh.visual.ColorVisuals()
        self.mesh.export(self.pathToMesh)
