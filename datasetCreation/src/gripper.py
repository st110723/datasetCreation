import os

import numpy as np
import yaml

from point import Point3D
from rectangle import AARectangle


class Gripper:

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        self.loadConfigFile()
        self.xRange = [-1, 1]
        self.yRange = [-1, 1]
        self.zRange = [0, 1]
        self.yawRange = [-np.pi, np.pi]
        self.xAxisID = -1
        self.yAxisID = -1
        self.zAxisID = -1

    def loadConfigFile(self):
        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.debug = cfg.get("debugConfig").get("debug")
        self.type = cfg.get("gripperConfig").get("type")
        if self.debug:
            if self.type not in ["robotiq85"]:
                raise ValueError("Invalid gripper type")
        self.pathToUrdf = cfg.get("gripperConfig").get("pathToUrdf")
        if self.debug:
            if not (os.path.exists(self.pathToUrdf)):
                raise FileNotFoundError
        self.startPosition = cfg.get("gripperConfig").get("startPosition")
        self.startOrientation = cfg.get("gripperConfig").get("startOrientation")
        self.lateralFriction = cfg.get("gripperConfig").get("lateralFriction")
        self.spinningFriction = cfg.get("gripperConfig").get("spinningFriction")
        self.rollingFriction = cfg.get("gripperConfig").get("rollingFriction")

    def load(self):
        self.ID = self.bulletClient.loadURDF(fileName=self.pathToUrdf,
                                             basePosition=self.startPosition,
                                             baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                 self.startOrientation))
        # flags=self.bulletClient.URDF_USE_SELF_COLLISION)

    @property
    def numJoints(self):
        return self.bulletClient.getNumJoints(self.ID)

    def get_joints_ids(self):
        jointDict = {}
        for jointID in range(self.numJoints):
            jointInfo = self.get_joint_info(jointID)
            jointName = jointInfo[1].decode("utf-8")
            jointDict[jointName] = jointID
        return jointDict

    def get_joint_info(self, jointID):
        return self.bulletClient.getJointInfo(self.ID, jointID)

    def get_joint_type(self, jointID):
        jointInfo = self.get_joint_info(jointID)
        return jointInfo[2]

    def get_joint_max_force(self, jointID):
        jointInfo = self.get_joint_info(jointID)
        return jointInfo[10]

    def get_joint_max_velocity(self, jointID):
        jointInfo = self.get_joint_info(jointID)
        return jointInfo[11]

    def set_joint_position(self, jointID, targetJointPosition):
        jointMaxForce = self.bulletClient.getJointInfo(self.ID, jointID)[10]
        jointMaxVelocity = self.bulletClient.getJointInfo(self.ID, jointID)[11]
        self.bulletClient.setJointMotorControl2(self.ID,
                                                jointID,
                                                self.bulletClient.POSITION_CONTROL,
                                                targetPosition=targetJointPosition,
                                                force=jointMaxForce,
                                                maxVelocity=jointMaxVelocity)

    def get_position(self):
        return self.bulletClient.getBasePositionAndOrientation(self.ID)[0]

    def draw_axis(self, dummyCenterLinkID):
        startPoint = self.bulletClient.getLinkState(self.ID, dummyCenterLinkID)[0]
        endPoint_x_axis = [startPoint[0] + 0.035, startPoint[1], startPoint[2]]
        self.xAxisID = self.bulletClient.addUserDebugLine(startPoint, endPoint_x_axis, [1, 0, 0], 0.5,
                                                          replaceItemUniqueId=self.xAxisID)
        endPoint_y_axis = [startPoint[0], startPoint[1] + 0.035, startPoint[2]]
        self.yAxisID = self.bulletClient.addUserDebugLine(startPoint, endPoint_y_axis, [0, 1, 0], 0.5,
                                                          replaceItemUniqueId=self.yAxisID)
        endPoint_z_axis = [startPoint[0], startPoint[1], startPoint[2] + 0.035]
        self.zAxisID = self.bulletClient.addUserDebugLine(startPoint, endPoint_z_axis, [0, 0, 1], 0.5,
                                                          replaceItemUniqueId=self.zAxisID)

    def add_user_control(self):
        return [self.bulletClient.addUserDebugParameter("x", self.xRange[0], self.xRange[1], 0),
                self.bulletClient.addUserDebugParameter("y", self.yRange[0], self.yRange[1], 0),
                self.bulletClient.addUserDebugParameter("z", self.zRange[0], self.zRange[1], 1),
                self.bulletClient.addUserDebugParameter("yaw", self.yawRange[0], self.yawRange[1], 0)]


class Robotiq85(Gripper):

    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        super().__init__(self.bulletClient)
        self.dummyCenterLinkID = 5
        self.leftFingerPadID = 8
        self.rightFingerPadID = 13
        self.openingAngleRange = [0, 0.8]

    @property
    def positionControlJointsIDs(self):
        joints = ["dummy_x_joint",
                  "dummy_y_joint",
                  "dummy_z_joint",
                  "dummy_yaw_joint"]
        jointsIDs = []
        for joint in joints:
            jointID = super().get_joints_ids()[joint]
            jointsIDs.append(jointID)
        return jointsIDs

    @property
    def openingControlJointID(self):
        joint = "finger_joint"
        jointID = super().get_joints_ids()[joint]
        return jointID

    @property
    def mimicJointsIDs(self):
        jointsDict = {"left_inner_knuckle_joint": 1,
                      "left_inner_finger_joint": -1,
                      "right_outer_knuckle_joint": 1,
                      "right_inner_knuckle_joint": 1,
                      "right_inner_finger_joint": -1}
        jointsIDs = {}
        for joint in jointsDict.keys():
            jointID = super().get_joints_ids()[joint]
            multiplier = jointsDict[joint]
            jointsIDs[jointID] = multiplier
        return jointsIDs

    def set_friction(self):
        self.bulletClient.changeDynamics(self.ID,
                                         self.leftFingerPadID,
                                         lateralFriction=self.lateralFriction,
                                         spinningFriction=self.spinningFriction,
                                         rollingFriction=self.rollingFriction,
                                         frictionAnchor=True)

        self.bulletClient.changeDynamics(self.ID,
                                         self.rightFingerPadID,
                                         lateralFriction=self.lateralFriction,
                                         spinningFriction=self.spinningFriction,
                                         rollingFriction=self.rollingFriction,
                                         frictionAnchor=True)

    def set_constraints(self):
        # for jointID in range(self.numJoints):
        #     jointType = self.get_joint_type(jointID)
        #     if jointType != self.bulletClient.JOINT_FIXED:
        #         self.bulletClient.setJointMotorControl2(self.gripperID, jointID,
        #         self.bulletClient.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for jointID, multiplier in self.mimicJointsIDs.items():
            c = self.bulletClient.createConstraint(self.ID,
                                                   self.openingControlJointID,
                                                   self.ID,
                                                   jointID,
                                                   jointType=self.bulletClient.JOINT_GEAR,
                                                   jointAxis=[0, 1, 0],
                                                   parentFramePosition=[0, 0, 0],
                                                   childFramePosition=[0, 0, 0])
            # TODO: check multiplier or -multiplier
            self.bulletClient.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def load(self):
        super().load()
        self.set_constraints()
        self.set_friction()

    def get_current_pose(self):
        currentPose = []
        for i, jointID in enumerate(self.positionControlJointsIDs):
            currentPose.append(self.bulletClient.getJointState(self.ID, jointID)[0])
        return currentPose

    def get_current_x(self):
        currentPose = self.get_current_pose()
        x = currentPose[0]
        return x

    def get_current_y(self):
        currentPose = self.get_current_pose()
        y = currentPose[1]
        return y

    def get_current_z(self):
        currentPose = self.get_current_pose()
        z = currentPose[2]
        return z

    def get_current_yaw(self):
        currentPose = self.get_current_pose()
        yaw = currentPose[3]
        return yaw

    def move(self, targetPose):
        assert len(targetPose) == 4, "Gripper pose must be a 4D array!"
        for i, jointID in enumerate(self.positionControlJointsIDs):
            self.set_joint_position(jointID, targetPose[i])

    def get_current_openingAngle(self):
        return self.bulletClient.getJointState(self.ID, self.openingControlJointID)[0]

    def open(self, openingAngle):
        self.set_joint_position(self.openingControlJointID, openingAngle)

    def close(self):
        self.bulletClient.setJointMotorControl2(self.ID,
                                                self.openingControlJointID,
                                                self.bulletClient.VELOCITY_CONTROL,
                                                targetVelocity=1,
                                                force=300)

    def fix_openingAngle(self, currentOpeningAngle):
        self.set_joint_position(self.openingControlJointID, currentOpeningAngle)

    def fix_pose(self, currentPose):
        assert len(currentPose) == 4, "Gripper pose must be a 4D array!"
        for i, jointID in enumerate(self.positionControlJointsIDs):
            self.set_joint_position(jointID, currentPose[i])

    def get_finger_bounding_box(self, fingerID):
        aabb = self.bulletClient.getAABB(self.ID, fingerID)
        z = aabb[0][2]
        bottomLeft = Point3D(aabb[0][0], aabb[0][1], z)
        topRight = Point3D(aabb[1][0], aabb[1][1], z)
        topLeft = Point3D(bottomLeft.x, topRight.y, z)
        bottomRight = Point3D(topRight.x, bottomLeft.y, z)
        boundingBox = AARectangle(topLeft, topRight, bottomRight, bottomLeft)
        return boundingBox

    def compute_opening_width(self):
        # Method 1
        leftFingerBoundingBox = self.get_finger_bounding_box(self.leftFingerPadID)
        rightFingerBoundingBox = self.get_finger_bounding_box(self.rightFingerPadID)

        pointLeftFinger = Point3D.from_np_array((leftFingerBoundingBox.topRight.to_np_array() +
                                                 leftFingerBoundingBox.bottomRight.to_np_array()) / 2)
        pointRightFinger = Point3D.from_np_array((rightFingerBoundingBox.topLeft.to_np_array() +
                                                  rightFingerBoundingBox.bottomLeft.to_np_array()) / 2)

        w1 = pointLeftFinger.distance_to(pointRightFinger)

        # Method 2
        currentOpeningAngle = self.get_current_openingAngle()
        w2 = 0.01 + 0.1143 * np.sin(0.715 - currentOpeningAngle)

        w = (w1 + w2) / 2
        return w

    def compute_jaw_size(self):
        leftFingerBoundingBox = self.get_finger_bounding_box(self.leftFingerPadID)
        rightFingerBoundingBox = self.get_finger_bounding_box(self.rightFingerPadID)
        h1 = leftFingerBoundingBox.height
        h2 = rightFingerBoundingBox.height
        h = (h1 + h2) / 2
        return h

    @staticmethod
    def worldCoordinates_to_gripperCoordinates(pointWorld):
        assert isinstance(pointWorld, Point3D), "Input must be a 3D point!"
        x = pointWorld.y
        y = pointWorld.x
        z = 1 - pointWorld.z
        pointGripper = Point3D(x, y, z)
        return pointGripper

    @staticmethod
    def theta_to_yaw(theta):
        yaw = -theta
        return yaw
