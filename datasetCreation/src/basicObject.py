import os
from copy import deepcopy

import numpy as np

from point import Point3D
from rectangle import AARectangle


class BasicObject:

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        self.pathToUrdf = None
        self.startPosition = None
        self.startOrientation = None
        self.lateralFriction = None
        self.spinningFriction = None
        self.rollingFriction = None
        self.mesh = None
        self.visualMesh = None
        self.collisionMesh = None
        self.urdf = None
        self.name = None

    def load(self):
        self.ID = self.bulletClient.loadURDF(fileName=self.pathToUrdf,
                                             basePosition=self.startPosition,
                                             baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                 self.startOrientation))
        # flags=self.bulletClient.URDF_USE_INERTIA_FROM_FILE)

    def set_friction(self):
        self.bulletClient.changeDynamics(self.ID, -1,
                                         lateralFriction=self.lateralFriction,
                                         spinningFriction=self.spinningFriction,
                                         rollingFriction=self.rollingFriction)

    def set_mass(self, mass):
        self.bulletClient.changeDynamics(self.ID, -1, mass=mass)

    def set_texture(self, color):
        # textureId = self.bulletClient.loadTexture("meshes/objects/rock.png")
        self.bulletClient.changeVisualShape(self.ID, -1, rgbaColor=color)

    def get_position_and_orientation(self):
        position_and_orientation = self.bulletClient.getBasePositionAndOrientation(self.ID)
        return position_and_orientation

    def get_position(self):
        position_and_orientation = self.get_position_and_orientation()
        position = position_and_orientation[0]
        return position

    def get_orientation(self):
        position_and_orientation = self.get_position_and_orientation()
        orientation = position_and_orientation[1]
        return orientation

    def get_euler_orientation(self):
        orientation = self.get_orientation()
        eulerOrientation = self.bulletClient.getEulerFromQuaternion(orientation)
        return eulerOrientation

    def set_position_and_orientation(self, position, orientation):
        self.bulletClient.resetBasePositionAndOrientation(self.ID, position, orientation)

    def remove(self):
        self.bulletClient.removeBody(self.ID)

    def generate_visual_mesh(self, scale):
        self.visualMesh = deepcopy(self.mesh)
        if scale:
            scaleFactor = self.visualMesh.get_scale_factor()
            self.visualMesh.scale(scaleFactor)
        self.visualMesh.change_origin(self.visualMesh.centerOfMass)
        filename = os.path.splitext(self.mesh.pathToMesh)[0]
        extension = os.path.splitext(self.mesh.pathToMesh)[1]
        self.visualMesh.pathToMesh = filename + "_visual" + extension
        self.visualMesh.save()

    def generate_collision_mesh(self):
        filename = os.path.splitext(self.mesh.pathToMesh)[0]
        extension = os.path.splitext(self.mesh.pathToMesh)[1]
        pathToCollisionMesh = filename + "_collision" + extension
        self.collisionMesh = self.mesh.decompose(self.visualMesh.pathToMesh, pathToCollisionMesh)

    def generate_urdf(self, scale=False):
        self.generate_visual_mesh(scale)
        self.generate_collision_mesh()
        self.urdf.set_name(self.name)
        self.urdf.set_visual_mesh_path(self.visualMesh.pathToMesh)
        self.urdf.set_collision_mesh_path(self.collisionMesh.pathToMesh)
        self.urdf.set_mass(self.collisionMesh.mass)
        self.urdf.set_inertia(self.collisionMesh.inertia)
        self.urdf.save(self.pathToUrdf)

    def get_bounding_box(self):
        mesh = self.bulletClient.getMeshData(self.ID, -1)[1]
        meshWorld = self.transform_mesh_to_worldCoordinates(mesh)
        bottomLeft = Point3D(np.min(np.array(meshWorld), axis=0)[0], np.min(np.array(meshWorld), axis=0)[1], 0)
        topRight = Point3D(np.max(np.array(meshWorld), axis=0)[0], np.max(np.array(meshWorld), axis=0)[1], 0)
        topLeft = Point3D(bottomLeft.x, topRight.y, 0)
        bottomRight = Point3D(topRight.x, bottomLeft.y, 0)
        return AARectangle(topLeft, topRight, bottomRight, bottomLeft)

    def transform_mesh_to_worldCoordinates(self, mesh):
        meshWorld = []
        for point in mesh:
            pointWorld = self.objectCoordinates_to_worldCoordinates(point)
            meshWorld.append(pointWorld)
        return meshWorld

    def objectCoordinates_to_worldCoordinates(self, point):
        position, quaternion = self.bulletClient.getBasePositionAndOrientation(self.ID)
        rotationMatrix = np.array(self.bulletClient.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        transformationMatrix = np.zeros((4, 4))
        transformationMatrix[:3, :3] = rotationMatrix
        transformationMatrix[:3, 3] = position
        transformationMatrix[3, 3] = 1
        objectCoordinates = np.append(np.array(point), 1)
        worldCoordinates = np.matmul(transformationMatrix, objectCoordinates)
        worldCoordinates = worldCoordinates[:-1]
        return worldCoordinates
