import csv
import logging
import os
import shutil
import time
import uuid

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import yaml
from scipy import ndimage

from camera import Camera
from gripper import Gripper, Robotiq85
from object import Object
from plane import Plane
from point import Point3D, Pixel
from rectangle import AARectangle, GraspingRectangle
from rock import Rock


class Simulation:

    def __init__(self):

        self.loadConfigFile()

        self.logFileName = str(uuid.uuid4())
        self.logger = self.setup_log_file(self.logFileName)
        self.logger.info("Start simulation")

        if self.debug:
            self.bulletClient = bc.BulletClient(connection_mode=pb.GUI,
                                                options="-background_color_red=0.64 "
                                                        "--background_color_green=0.47 "
                                                        "--background_color_blue=0.23 "
                                                        "--width=1366 "
                                                        "--height=706")
        else:
            self.bulletClient = bc.BulletClient(connection_mode=pb.DIRECT)

        self.plane = Plane(bulletClient=self.bulletClient)
        self.object = Object(bulletClient=self.bulletClient)
        self.numRocks = np.asscalar(np.random.randint(5, 21, size=1))
        self.logger.debug(f"Number of rocks: {self.numRocks}")
        self.rocks = [Rock(bulletClient=self.bulletClient) for _ in range(self.numRocks)]
        self.gripper = Gripper(bulletClient=self.bulletClient)
        if self.gripper.type == "robotiq85":
            self.gripper = Robotiq85(bulletClient=self.bulletClient)
        self.camera = Camera(bulletClient=self.bulletClient)
        self.textContactPlaneID = -1
        self.textContactObjectID = -1
        self.ID = None
        self.objectInitialPosition = None
        self.objectInitialOrientation = None

    def loadConfigFile(self):
        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.debug = cfg.get("debugConfig").get("debug")
        self.rgbDir = cfg.get("dirConfig").get("rgbDir")
        self.depthDir = cfg.get("dirConfig").get("depthDir")
        self.segmentationDir = cfg.get("dirConfig").get("segmentationDir")
        self.rgbWithLabelDir = cfg.get("dirConfig").get("rgbWithLabelDir")
        self.csvDir = cfg.get("dirConfig").get("csvDir")
        self.logsDir = cfg.get("dirConfig").get("logsDir")
        self.timeStep = cfg.get("simConfig").get("timeStep")
        self.numIterations = cfg.get("simConfig").get("numIterations")
        self.gravity = cfg.get("simConfig").get("gravity")
        self.mode = cfg.get("simConfig").get("mode")
        if self.debug:
            if self.mode not in ["manual", "auto"]:
                raise ValueError("Invalid simulation mode")
        self.epsilon = cfg.get("simConfig").get("epsilon")
        self.enableGUI = cfg.get("simConfig").get("enableGUI")
        self.createVideo = cfg.get("simConfig").get("createVideo")
        if self.createVideo:
            self.pathToVideo = cfg.get("simConfig").get("pathToVideo")

    def setup_log_file(self, name):
        logger = logging.getLogger(name)
        handler = logging.FileHandler(f"{self.logsDir}/{name}.log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger

    def init(self):
        self.set_gravity()
        self.set_time_step()
        self.set_number_of_iterations()
        self.configure_visualizer(cameraDistance=1.5, cameraYaw=20, cameraPitch=-25, cameraTargetPosition=[0, 0, 0])
        if self.createVideo:
            self.videoID = self.bulletClient.startStateLogging(self.bulletClient.STATE_LOGGING_VIDEO_MP4,
                                                               self.pathToVideo)

    def set_gravity(self):
        self.bulletClient.setGravity(0, 0, self.gravity)

    def set_time_step(self):
        self.bulletClient.setTimeStep(self.timeStep)

    def set_number_of_iterations(self):
        self.bulletClient.setPhysicsEngineParameter(numSolverIterations=self.numIterations)

    def configure_visualizer(self, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition):
        self.bulletClient.configureDebugVisualizer(self.bulletClient.COV_ENABLE_GUI, self.enableGUI)
        self.bulletClient.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                                                     cameraYaw=cameraYaw,
                                                     cameraPitch=cameraPitch,
                                                     cameraTargetPosition=cameraTargetPosition)

    def load_plane(self):
        if self.debug:
            self.print_text("Loading plane", [0.1, 0.1, 1])
        self.logger.info("Loading plane")
        self.plane.load()

    def load_object(self):
        objectOutsideFOV = True
        counter = 0
        while objectOutsideFOV:
            counter += 1
            if counter > 1:
                self.object.remove()
            self.drop_object()
            if self.camera.position == "above_object":
                self.set_camera_position_above_object()
            self.camera.get_image()
            objectBoundingBox = self.object.get_bounding_box()
            cameraFOV = self.camera.get_field_of_view()
            if self.debug:
                self.print_text("Checking if object is outside camera FOV", [0.1, 0.1, 1])
                objectBoundingBox.draw_in_gui(self.bulletClient, displayDuration=0)
                cameraFOV.draw_in_gui(self.bulletClient, displayDuration=0)
            self.logger.info("Checking if object is outside camera FOV")
            objectOutsideFOV = not (cameraFOV.contains(objectBoundingBox))
            self.logger.debug(f"Object outside camera FOV: {objectOutsideFOV}")
            if self.debug:
                if objectOutsideFOV:
                    self.print_text("Object outside camera FOV!", [0.1, 0.1, 1])
                else:
                    self.print_text("Object inside camera FOV!", [0.1, 0.1, 1])
            self.ID = self.generate_simulation_ID()
            self.objectInitialPosition = self.object.get_position()
            self.objectInitialOrientation = self.object.get_orientation()
            self.bulletClient.removeAllUserDebugItems()

    def set_camera_position_above_object(self):
        objectPosition = self.object.get_position()
        self.camera.eyePosition = np.array([objectPosition[0], objectPosition[1], 0.5])
        self.camera.targetPosition = np.array([objectPosition[0], objectPosition[1], 0])

    def generate_simulation_ID(self):
        h = hash(self.object.get_position_and_orientation())
        ID = str(h) if h >= 0 else str(h).replace("-", "m")
        return ID

    def get_camera_image(self):
        self.print_text("Getting camera image", [0.1, 0.1, 1])
        self.logger.info("Getting camera image")
        self.camera.get_image()
        imageName = self.ID
        self.logger.debug(f"Image name: {imageName}")
        self.camera.save_rgb_image(f"{self.rgbDir}/{imageName}.png")
        shutil.copyfile(f"{self.rgbDir}/{imageName}.png", f"{self.rgbWithLabelDir}/failed_grasps/{imageName}.png")
        shutil.copyfile(f"{self.rgbDir}/{imageName}.png", f"{self.rgbWithLabelDir}/successful_grasps/{imageName}.png")
        self.camera.save_segmentation_mask(self.object.ID, f"{self.segmentationDir}/{imageName}.png")
        self.camera.save_depth_image(f"{self.depthDir}/{imageName}.tiff")

    def drop_object(self):
        if self.debug:
            self.print_text("Loading object", [0.1, 0.1, 1])
        self.logger.info("Loading object")
        self.object.load()
        self.logger.info(f"Object start orientation: {self.object.startOrientation}")
        if self.debug:
            self.print_text("Dropping object", [0.1, 0.1, 1])
        self.logger.info("Dropping object")
        counter, delta = self.drop(self.object)
        self.logger.debug(f"Counter: {counter}, Delta: {delta}")
        return counter, delta

    def drop(self, object):
        still = False
        counter = 0
        previousPosition = object.get_position()
        delta = None
        while (not still) and (counter < 1000):
            counter += 1
            self.step_simulation()
            currentPosition = object.get_position()
            delta = np.linalg.norm(np.array(previousPosition) - np.array(currentPosition))
            previousPosition = currentPosition
            still = delta < 1e-10
        return counter, delta

    def step_simulation(self):
        self.bulletClient.stepSimulation()
        if self.createVideo:
            time.sleep(10 * self.timeStep)

    def load_rocks(self):
        if self.debug:
            self.print_text("Loading rocks", [0.1, 0.1, 1])
        self.logger.info("Loading rocks")
        objectBoundingBox = self.object.get_bounding_box()
        objectBoundingBox.enlarge(0.2)
        objectBoundingBox.draw_in_gui(self.bulletClient, displayDuration=0)
        for i in range(self.numRocks):
            rockFarFromObject = False
            counter = 0
            while not rockFarFromObject:
                counter += 1
                if counter > 1:
                    self.bulletClient.removeBody(self.rocks[i].ID)
                self.rocks[i].startPosition = np.append(np.random.default_rng().uniform(-1, 1, 2), 1)
                self.rocks[i].startOrientation = [0, 0, np.asscalar(np.random.default_rng().uniform(-np.pi, np.pi, 1))]
                self.rocks[i].load()
                rockBoundingBox = self.rocks[i].get_bounding_box()
                rockBoundingBox.draw_in_gui(self.bulletClient, displayDuration=0)
                rockFarFromObject = rockBoundingBox.is_outside(objectBoundingBox)
            self.drop(self.rocks[i])
        self.bulletClient.removeAllUserDebugItems()

    def load_gripper(self):
        if self.debug:
            self.print_text("Loading gripper", [0.1, 0.1, 1])
        self.logger.info("Loading gripper")
        self.gripper.load()
        if self.debug:
            self.print_text("Moving gripper to start position", [0.1, 0.1, 1])
        self.logger.info("Moving gripper to start position")
        counter, error = self.move_gripper_to_start_position()
        self.logger.debug(f"Counter: {counter}, Error: {error}")

    def move_gripper_to_start_position(self):
        counter, error, _ = self.move_gripper([0, 0, 0.8, 0], checkContact=0)
        return counter, error

    def get_contact_plane(self):
        return self.bulletClient.getContactPoints(self.gripper.ID, self.plane.ID)

    def get_contact_object(self):
        return self.bulletClient.getContactPoints(self.gripper.ID, self.object.ID)

    def check_contact_plane(self):
        contactPoints = self.get_contact_plane()
        contactPlane = 0
        if len(contactPoints) > 0:
            for point in contactPoints:
                normalForce = point[9]
                if normalForce > 0:
                    contactPlane = 1
                    break
        return contactPlane

    def check_contact_object(self):
        contactPoints = self.get_contact_object()
        contactObject = 0
        normalForce = 0
        if len(contactPoints) > 0:
            for point in contactPoints:
                penetration = point[8]
                normalForce += point[9]
            normalForce /= len(contactPoints)
            # contactObject = normalForce > 0
            contactObject = 1
        return contactObject

    def print_contact(self):
        startPoint = self.bulletClient.getLinkState(self.gripper.ID, self.gripper.dummyCenterLinkID)[0]
        textContactPlanePosition = [startPoint[0] + 0.1, startPoint[1] + 0.1, startPoint[2]]
        contactPlane = self.check_contact_plane()
        if contactPlane:
            contactPoints = self.get_contact_plane()
            self.textContactPlaneID = self.bulletClient.addUserDebugText(
                f"Contact with plane! Num of contact points = {len(contactPoints)}",
                textContactPlanePosition,
                [0, 0, 0],
                replaceItemUniqueId=self.textContactPlaneID)
        else:
            self.textContactPlaneID = self.bulletClient.addUserDebugText(f"No contact with plane!",
                                                                         textContactPlanePosition,
                                                                         [0, 0, 0],
                                                                         replaceItemUniqueId=self.textContactPlaneID)
        textContactObjectPosition = [startPoint[0] + 0.1, startPoint[1] + 0.1, startPoint[2] + 0.1]
        contactObject = self.check_contact_object()
        if contactObject:
            contactPoints = self.get_contact_object()
            self.textContactObjectID = self.bulletClient.addUserDebugText(
                f"Contact with object! Num of contact points = {len(contactPoints)}",
                textContactObjectPosition,
                [0, 0, 0],
                replaceItemUniqueId=self.textContactObjectID)
        else:
            self.textContactObjectID = self.bulletClient.addUserDebugText(f"No contact with object!",
                                                                          textContactObjectPosition,
                                                                          [0, 0, 0],
                                                                          replaceItemUniqueId=self.textContactObjectID)

    def check_grasp_success(self):
        contactPointsLeft = self.bulletClient.getContactPoints(self.gripper.ID, self.object.ID,
                                                               self.gripper.leftFingerPadID)
        contactLeft = len(contactPointsLeft) > 0
        contactPointsRight = self.bulletClient.getContactPoints(self.gripper.ID, self.object.ID,
                                                                self.gripper.rightFingerPadID)
        contactRight = len(contactPointsRight) > 0
        graspSuccessful = contactLeft and contactRight
        return graspSuccessful

    def print_grasp(self):
        graspSuccessful = self.check_grasp_success()
        if graspSuccessful:
            self.print_text("Grasp Successful!", [0.1, 0.1, 0.2])
        else:
            self.print_text("Grasp failed!", [0.1, 0.1, 0.2])

    def print_text(self, text, position):
        textID = self.bulletClient.addUserDebugText(text, position, [0, 0, 0], lifeTime=1000 * self.timeStep)
        time.sleep(1)
        self.bulletClient.removeUserDebugItem(textID)

    def get_object_bounding_box_from_image(self):
        segmentationMask = self.camera.segmentationMask
        objectsList = ndimage.measurements.find_objects(segmentationMask == self.object.ID)
        assert len(objectsList) == 1, "More than one object detected!"
        object = objectsList[0]

        # Bottom left corner
        iBottomLeft, jBottomLeft = object[0].stop - 1, object[1].start
        uBottomLeft, vBottomLeft = jBottomLeft, iBottomLeft
        bottomLeft = Pixel(uBottomLeft, vBottomLeft)

        # Bottom right corner
        iBottomRight, jBottomRight = object[0].stop - 1, object[1].stop - 1
        uBottomRight, vBottomRight = jBottomRight, iBottomRight
        bottomRight = Pixel(uBottomRight, vBottomRight)

        # Top right corner
        iTopRight, jTopRight = object[0].start, object[1].stop - 1
        uTopRight, vTopRight = jTopRight, iTopRight
        topRight = Pixel(uTopRight, vTopRight)

        # Top left corner
        iTopLeft, jTopLeft = object[0].start, object[1].start
        uTopLeft, vTopLeft = jTopLeft, iTopLeft
        topLeft = Pixel(uTopLeft, vTopLeft)
        boundingBox = AARectangle(topLeft, topRight, bottomRight, bottomLeft)
        return boundingBox

    def move_gripper(self, targetPose, checkContact):
        self.gripper.move(targetPose)
        currentPose = self.gripper.get_current_pose()
        error = np.linalg.norm(np.abs(np.array(currentPose) - np.array(targetPose)))
        counter = 0
        contact = None
        while error > self.epsilon:
            self.bulletClient.stepSimulation()
            counter += 1
            currentPose = self.gripper.get_current_pose()
            error = np.linalg.norm(np.abs(np.array(currentPose) - np.array(targetPose)))
            if checkContact:
                # contact = self.check_contact_plane() or self.check_contact_object()
                contact = self.check_contact_object()
                if contact:
                    break
        self.gripper.fix_pose(currentPose)
        return counter, error, contact

    def open_gripper(self, targetOpeningAngle, checkContact):
        self.gripper.open(targetOpeningAngle)
        currentOpeningAngle = self.gripper.get_current_openingAngle()
        error = np.abs(currentOpeningAngle - targetOpeningAngle)
        counter = 0
        contact = None
        while (error > self.epsilon) and (counter < 1000):
            self.gripper.set_constraints()
            self.bulletClient.stepSimulation()
            counter += 1
            currentOpeningAngle = self.gripper.get_current_openingAngle()
            error = np.abs(currentOpeningAngle - targetOpeningAngle)
            if checkContact:
                contact = self.check_contact_plane() or self.check_contact_object()
                if contact:
                    break
        self.gripper.fix_openingAngle(currentOpeningAngle)
        return counter, error, contact

    def close_gripper(self, checkContact):
        self.gripper.close()
        currentOpeningAngle = self.gripper.get_current_openingAngle()
        error = np.abs(currentOpeningAngle - self.gripper.openingAngleRange[1])
        counter = 0
        contact = None
        while (error > self.epsilon) and (counter < 2000):
            self.gripper.set_constraints()
            self.bulletClient.stepSimulation()
            counter += 1
            currentOpeningAngle = self.gripper.get_current_openingAngle()
            error = np.abs(currentOpeningAngle - self.gripper.openingAngleRange[1])
            if checkContact:
                contact = self.check_contact_object()
                # if contact:
                #     break
        self.gripper.fix_openingAngle(currentOpeningAngle)
        return counter, error, contact

    def compute_opening_width_in_pixel(self):
        leftFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.leftFingerPadID)
        rightFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.rightFingerPadID)

        pointLeftFinger = Point3D.from_np_array((leftFingerBoundingBox.topRight.to_np_array() +
                                                 leftFingerBoundingBox.bottomRight.to_np_array()) / 2)
        pointRightFinger = Point3D.from_np_array((rightFingerBoundingBox.topLeft.to_np_array() +
                                                  rightFingerBoundingBox.bottomLeft.to_np_array()) / 2)

        if self.debug:
            leftFingerBoundingBox.draw_in_gui(self.bulletClient, displayDuration=1000 * self.timeStep)
            rightFingerBoundingBox.draw_in_gui(self.bulletClient, displayDuration=1000 * self.timeStep)
            self.bulletClient.addUserDebugLine(pointRightFinger.to_list(), pointLeftFinger.to_list(), [0, 0, 0], 2,
                                               lifeTime=1000 * self.timeStep)
            time.sleep(1)
            self.bulletClient.removeAllUserDebugItems()

        pixelLeftFinger, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(pointLeftFinger)

        pixelRightFinger, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(pointRightFinger)

        w = pixelLeftFinger.distance_to(pixelRightFinger)
        return w

    def get_grasping_rectangle(self):
        leftFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.leftFingerPadID)
        rightFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.rightFingerPadID)
        topLeft = leftFingerBoundingBox.topRight
        topRight = rightFingerBoundingBox.topLeft
        bottomRight = rightFingerBoundingBox.bottomLeft
        bottomLeft = leftFingerBoundingBox.bottomRight
        graspingRectangle = AARectangle(topLeft, topRight, bottomRight, bottomLeft)
        return graspingRectangle

    def get_grasping_rectangle_in_pixel(self):
        leftFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.leftFingerPadID)
        rightFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.rightFingerPadID)
        topLeft, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.topRight)
        topRight, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.topLeft)
        bottomRight, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.bottomLeft)
        bottomLeft, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.bottomRight)
        graspingRectangle = AARectangle(topLeft, topRight, bottomRight, bottomLeft)
        return graspingRectangle

    def compute_jaw_size_in_pixel(self):
        leftFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.leftFingerPadID)
        rightFingerBoundingBox = self.gripper.get_finger_bounding_box(self.gripper.rightFingerPadID)
        pixel1, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.topLeft)
        pixel2, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.bottomLeft)
        h1 = pixel1.distance_to(pixel2)

        pixel1, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.topRight)
        pixel2, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(leftFingerBoundingBox.bottomRight)
        h2 = pixel1.distance_to(pixel2)

        pixel1, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.topLeft)
        pixel2, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.bottomLeft)
        h3 = pixel1.distance_to(pixel2)

        pixel1, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.topRight)
        pixel2, _ = self.camera.worldCoordinates_to_pixel_depthBuffer(rightFingerBoundingBox.bottomRight)
        h4 = pixel1.distance_to(pixel2)

        h = (h1 + h2 + h3 + h4) / 4
        return h

    def reset(self):
        self.object.set_position_and_orientation(self.objectInitialPosition, self.objectInitialOrientation)
        self.move_gripper_to_start_position()

    def step(self):
        if self.mode == "manual":
            # self.gripper.open(self.gripper.openingAngleRange[0])
            userControlParameters = self.gripper.add_user_control()

            while True:
                # Add gripper axis
                self.gripper.draw_axis(self.gripper.dummyCenterLinkID)

                # Gripper pose control
                xWorld, yWorld, zWorld = [self.bulletClient.readUserDebugParameter(userControlParameters[0]),
                                          self.bulletClient.readUserDebugParameter(userControlParameters[1]),
                                          self.bulletClient.readUserDebugParameter(userControlParameters[2])]
                poseWorld = Point3D(xWorld, yWorld, zWorld)
                poseGripper = self.gripper.worldCoordinates_to_gripperCoordinates(poseWorld)
                targetGripperPose = poseGripper.to_list()
                yaw = self.bulletClient.readUserDebugParameter(userControlParameters[3])
                targetGripperPose.append(yaw)
                counter, error, contact = self.move_gripper(targetGripperPose, checkContact=False)

                # Contact
                self.print_contact()

                # Gripper opening control
                keys = self.bulletClient.getKeyboardEvents()
                # C => close gripper
                if 99 in keys:
                    counter, error, contact = self.close_gripper(checkContact=0)
                # O => open gripper
                elif 111 in keys:
                    counter, error, contact = self.open_gripper(self.gripper.openingAngleRange[0])

        elif self.mode == "auto":
            logging.shutdown()
            os.rename(f"{self.logsDir}/{self.logFileName}.log", f"{self.logsDir}/{self.ID}.log")

            for i in range(100):
                # Initialization
                graspID = f"{self.ID}_{i + 1}"
                logger = self.setup_log_file(graspID)
                logger.info(f"Grasp attempt: {i + 1}")

                if i == 0:
                    self.configure_visualizer(cameraDistance=0.5, cameraYaw=20, cameraPitch=-25,
                                              cameraTargetPosition=self.object.get_position())
                else:
                    self.reset()

                # Open gripper
                if self.debug:
                    self.print_text("Sampling opening angle", [0.1, 0.1, 0.2])
                logger.info("Sampling opening angle")

                targetOpeningAngle = 0
                # targetOpeningAngle = np.asscalar(np.random.default_rng().uniform(self.gripper.openingAngleRange[0],
                #                                                                  0.4, #self.gripper.openingAngleRange[1],
                #                                                                  1))
                logger.debug(f"Opening angle: {targetOpeningAngle}")

                if self.debug:
                    self.print_text("Opening gripper", [0.1, 0.1, 0.2])
                logger.info("Opening gripper")
                counter, error, _ = self.open_gripper(targetOpeningAngle, checkContact=False)
                logger.debug(f"Counter: {counter}, Error:{error}")

                # Get object bounding box from image
                boundingBox = self.get_object_bounding_box_from_image()

                # Move gripper
                if self.debug:
                    self.print_text("Sampling gripper position", [0.1, 0.1, 0.2])
                logger.info("Sampling gripper position")
                # targetPixel = Pixel(boundingBox.center.x, boundingBox.center.y)
                # u, v = np.random.multivariate_normal(boundingBox.center.to_np_array(),
                #                                      np.diag([boundingBox.height/2, boundingBox.width/2]))
                vList, uList = np.where(self.camera.segmentationMask == self.object.ID)
                objectPixels = list(zip(uList, vList))
                u, v = objectPixels[np.random.choice(len(objectPixels))]
                targetPixel = Pixel(int(u), int(v))
                d = np.min(self.camera.depthBuffer)
                logger.debug(f"u: {u}, v: {v}, d:{d}")
                poseWorld = self.camera.pixel_depthBuffer_to_worldCoordinates(targetPixel, d)
                logger.debug(f"x_world: {poseWorld.x}, y_world: {poseWorld.y}, z_world:{poseWorld.z}")
                poseGripper = self.gripper.worldCoordinates_to_gripperCoordinates(poseWorld)
                targetGripperPose = poseGripper.to_list()
                yaw = self.gripper.get_current_yaw()
                targetGripperPose.append(yaw)
                logger.debug(f"Gripper pose: {targetGripperPose}")
                if self.debug:
                    self.print_text("Moving gripper", [0.1, 0.1, 0.2])
                logger.info("Moving gripper")
                counter, error, _ = self.move_gripper(targetGripperPose, checkContact=0)
                logger.debug(f"Counter: {counter}, Error:{error}")

                # Compute opening width and jaw size
                if self.debug:
                    self.print_text("Computing opening width and jaw size", [0.1, 0.1, 0.2])
                logger.info("Computing opening width and jaw size")
                w = self.compute_opening_width_in_pixel()
                h = self.compute_jaw_size_in_pixel()
                logger.debug(f"w: {w}, h: {h}")

                # Get grasping rectangle before rotation
                if self.debug:
                    graspingRectangleGUI = self.get_grasping_rectangle()
                    graspingRectangleGUI.draw_in_gui(self.bulletClient, displayDuration=1000 * self.timeStep)
                graspingRectangleInPixel = self.get_grasping_rectangle_in_pixel()
                graspingRectangleInPixel.draw_in_image(f"{self.rgbDir}/{self.ID}.png",
                                                       f"{self.rgbWithLabelDir}/{graspID}.png")

                # Rotate Gripper
                x = self.gripper.get_current_x()
                y = self.gripper.get_current_y()
                z = self.gripper.get_current_z()
                if self.debug:
                    self.print_text("Sampling gripper rotation angle", [0.1, 0.1, 0.2])
                logger.info("Sampling gripper rotation angle")
                theta = 0
                # theta = np.asscalar(np.random.default_rng().uniform(-np.pi / 2, np.pi / 2, 1))
                objectYaw = self.object.get_euler_orientation()[2]
                logger.debug(f"Object yaw: {objectYaw}")
                if -np.pi/2 <= objectYaw <= np.pi/2:
                    theta0 = objectYaw
                elif np.pi/2 < objectYaw <= np.pi:
                    theta0 = objectYaw - np.pi
                elif -np.pi <= objectYaw < -np.pi/2:
                    theta0 = np.pi + objectYaw
                else:
                    raise ValueError("Object yaw outside the interval [-pi, pi]!")
                logger.debug(f"Theta0: {theta0}")
                # theta = np.asscalar(np.random.default_rng().uniform(theta0-np.deg2rad(20), theta0+np.deg2rad(20), 1))
                # theta = np.asscalar(np.random.normal(theta0, np.deg2rad(10), 1))
                logger.debug(f"Theta: {theta}")
                yaw = self.gripper.theta_to_yaw(theta)
                if self.debug:
                    self.print_text("Rotating gripper", [0.1, 0.1, 0.2])
                logger.info("Rotating gripper")
                targetGripperPose = [x, y, z, yaw]
                logger.debug(f"Gripper pose: {targetGripperPose}")
                counter, error, _ = self.move_gripper(targetGripperPose, checkContact=0)
                logger.debug(f"Counter: {counter}, Error:{error}")

                # Get grasping rectangle after rotation
                if self.debug:
                    graspingRectangleGUI.rotate_in_gui(theta)
                    graspingRectangleGUI.draw_in_gui(self.bulletClient, displayDuration=1000 * self.timeStep)

                graspingRectangleInPixel.rotate_in_image(theta)
                graspingRectangleInPixel.draw_in_image(f"{self.rgbWithLabelDir}/{graspID}.png",
                                                       f"{self.rgbWithLabelDir}/{graspID}.png")

                # Move gripper downwards
                # if self.debug:
                #     self.print_text("Moving gripper downwards until contact", [0.1, 0.1, 0.2])
                # logger.info("Moving gripper downwards until contact")
                # targetGripperPose = [x, y, 1, yaw]
                # counter, error, contact = self.move_gripper(targetGripperPose, checkContact=1)
                # logger.debug(f"Counter: {counter}, Error: {error}, Contact:{contact}")
                #
                # Close gripper
                # if self.debug:
                #     self.print_text("Closing gripper", [0.1, 0.1, 0.2])
                # logger.info("Closing gripper")
                # counter, error, _ = self.close_gripper(checkContact=0)
                # logger.debug(f"Counter: {counter}, Error: {error}")
                #
                # Move gripper upwards
                # if self.debug:
                #     self.print_text("Moving gripper upwards", [0.1, 0.1, 0.2])
                # logger.info("Moving gripper upwards")
                # targetGripperPose = [x, y, 0.8, yaw]
                # counter, error, _ = self.move_gripper(targetGripperPose, checkContact=0)
                # logger.debug(f"Counter: {counter}, Error: {error}")
                #
                # Check if grasp is successful
                # graspSuccessful = self.check_grasp_success()
                # if self.debug:
                #     self.print_grasp()
                # logger.debug(f"Grasp successful: {graspSuccessful}")
                # logger.info("Saving data")
                # logging.shutdown()
                #
                # Save images with grasping rectangles
                # graspingRectangle = GraspingRectangle(targetPixel, w, h, theta, graspSuccessful)
                # if graspSuccessful:
                #     folderName = "successful_grasps"
                # else:
                #     folderName = "failed_grasps"
                #
                # graspingRectangle.draw_in_image(f"{self.rgbWithLabelDir}/{folderName}/{self.ID}.png",
                #                                 f"{self.rgbWithLabelDir}/{folderName}/{self.ID}.png")
                # os.replace(f"{self.rgbWithLabelDir}/{graspID}.png",
                #            f"{self.rgbWithLabelDir}/{folderName}/{graspID}.png")
                #
                # Save data
                # csvHeader = ["img", "x", "y", "w", "h", "theta"]
                # csvFilePath = f"{self.csvDir}/{folderName}/{self.ID}.csv"
                # csvFileExists = os.path.isfile(csvFilePath)
                # with open(csvFilePath, "a", newline="") as csvFile:
                #     csvWriter = csv.writer(csvFile, delimiter=",", lineterminator="\n")
                #     if not csvFileExists:
                #         csvWriter.writerow(csvHeader)
                #     csvWriter.writerow([self.ID, targetPixel.u, targetPixel.v, w, h, theta])

            if self.createVideo:
                self.bulletClient.stopStateLogging(self.videoID)
        else:
            raise NotImplemented
