import cv2
import glm
import numpy as np
import yaml

from point import Point3D, Point2D, Pixel
from rectangle import AARectangle


class Camera:
    def __init__(self, bulletClient):
        self.bulletClient = bulletClient
        self.loadConfigFile()
        self.eyePosition = None
        self.targetPosition = None

    def loadConfigFile(self):

        with open("src/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.debug = cfg.get("debugConfig").get("debug")
        self.imageWidth = cfg.get("cameraConfig").get("imageWidth")
        self.imageHeight = cfg.get("cameraConfig").get("imageHeight")
        # TODO: source vertical FOV for OpenGL
        self.fov = cfg.get("cameraConfig").get("fov")
        self.aspect = cfg.get("cameraConfig").get("aspect")
        self.nearVal = cfg.get("cameraConfig").get("nearVal")
        self.farVal = cfg.get("cameraConfig").get("farVal")
        self.position = cfg.get("cameraConfig").get("position")
        if self.debug:
            if self.position not in ["above_object", "on_gripper"]:
                raise ValueError("Invalid camera position")

    @property
    def viewMatrix(self):
        return self.bulletClient.computeViewMatrix(cameraEyePosition=self.eyePosition,
                                                   cameraTargetPosition=self.targetPosition,
                                                   cameraUpVector=[0, 1, 0])

    @property
    def projectionMatrix(self):
        return self.bulletClient.computeProjectionMatrixFOV(fov=self.fov,
                                                            aspect=self.aspect,
                                                            nearVal=self.nearVal,
                                                            farVal=self.farVal)

    def get_image(self):
        self.image = self.bulletClient.getCameraImage(width=self.imageWidth,
                                                      height=self.imageHeight,
                                                      viewMatrix=self.viewMatrix,
                                                      projectionMatrix=self.projectionMatrix)

    @property
    def rgbImage(self):
        if self.image is not None:
            rgbAlphaImage = np.array(self.image[2]).reshape((self.imageHeight, self.imageWidth, 4))
            rgbImage = rgbAlphaImage[:, :, :3]
            return rgbImage
        else:
            return None

    @property
    def depthBuffer(self):
        if self.image is not None:
            depthBuffer = np.array(self.image[3]).reshape((self.imageHeight, self.imageWidth))
            return depthBuffer
        else:
            return None

    def depthBuffer_to_depthImage(self, depthBuffer):
        depthImage = self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depthBuffer)
        return depthImage

    @property
    def depthImage(self):
        # TODO add noise to depth image
        if self.image is not None:
            depthImage = self.depthBuffer_to_depthImage(self.depthBuffer)
            return depthImage
        else:
            return None

    @property
    def segmentationMask(self):
        if self.image is not None:
            segmentationMask = np.array(self.image[4]).reshape((self.imageHeight, self.imageWidth))
            return segmentationMask
        else:
            return None

    def pixel_depthBuffer_to_NDC(self, pixel, d):
        assert isinstance(pixel, Pixel), "Input must be a pixel!"
        x = (2 / self.imageWidth) * pixel.u - 1
        y = (-2 / self.imageHeight) * pixel.v + 1
        z = 2 * d - 1
        NDC = Point3D(x, y, z)
        return NDC

    def pixel_to_windowCoordinates(self, pixel):
        assert isinstance(pixel, Pixel), "Input must be a pixel!"
        x = pixel.u
        y = self.imageHeight - pixel.v
        windowCoordinates = Point2D(x, y)
        return windowCoordinates

    def windowCoordinates_to_pixel(self, windowCoordinates):
        assert isinstance(windowCoordinates, Point2D), "Input must be a 2D Point!"
        u = windowCoordinates.x
        v = self.imageHeight - windowCoordinates.y
        pixel = Pixel(u, v)
        return pixel

    def NDC_to_worldCoordinates(self, NDC):
        assert isinstance(NDC, Point3D), "Input must be a 3D Point!"
        NDC = np.array([NDC.x, NDC.y, NDC.z, 1])
        projectionMatrix = np.array(self.projectionMatrix).reshape((4, 4), order="F")
        viewMatrix = np.array(self.viewMatrix).reshape((4, 4), order="F")
        transformationMatrix = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        worldCoordinates = np.matmul(transformationMatrix, NDC)
        worldCoordinates /= worldCoordinates[3]
        return Point3D.from_np_array(worldCoordinates[:3])

    def pixel_depthBuffer_to_worldCoordinates(self, pixel, d):
        assert isinstance(pixel, Pixel), "Input must be a pixel!"
        win = glm.vec3(self.pixel_to_windowCoordinates(pixel).to_tuple(), d)
        viewport = glm.vec4(0, 0, self.imageWidth, self.imageHeight)
        projectionMatrix = glm.mat4(np.array(self.projectionMatrix).reshape((4, 4), order="F"))
        viewMatrix = glm.mat4(np.array(self.viewMatrix).reshape((4, 4), order="F"))
        x, y, z = glm.unProject(win, viewMatrix, projectionMatrix, viewport)
        worldCoordinates = Point3D(x, y, z)
        return worldCoordinates

        # xNDC, yNDC = self.pixel_to_NDC(u, v)
        # zNDC = self.depthBuffer_to_NDC(d)
        # xWorld, yWorld, zWorld = self.NDC_to_worldCoordinates(xNDC, yNDC, zNDC)
        # return xWorld, yWorld, zWorld

    def get_field_of_view(self):
        bottomLeft = self.pixel_depthBuffer_to_worldCoordinates(Pixel(0, self.imageHeight), 1)
        bottomRight = self.pixel_depthBuffer_to_worldCoordinates(Pixel(self.imageWidth, self.imageHeight), 1)
        topLeft = self.pixel_depthBuffer_to_worldCoordinates(Pixel(0, 0), 1)
        topRight = self.pixel_depthBuffer_to_worldCoordinates(Pixel(self.imageWidth, 0), 1)
        fov = AARectangle(topLeft, topRight, bottomRight, bottomLeft)
        return fov

    def worldCoordinates_to_pixel_depthBuffer(self, point):
        assert isinstance(point, Point3D), "Input must be a 3D Point!"
        obj = glm.vec3(point.x, point.y, point.z)
        viewport = glm.vec4(0, 0, self.imageWidth, self.imageHeight)
        projectionMatrix = glm.mat4(np.array(self.projectionMatrix).reshape((4, 4), order="F"))
        viewMatrix = glm.mat4(np.array(self.viewMatrix).reshape((4, 4), order="F"))
        xWindow, yWindow, d = glm.project(obj, viewMatrix, projectionMatrix, viewport)
        pixel = self.windowCoordinates_to_pixel(Point2D(xWindow, yWindow))
        return pixel, d

    def save_rgb_image(self, path):
        # TODO add alpha channel?
        img = self.rgbImage.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def save_segmentation_mask(self, objectID, path):
        img = np.array(self.segmentationMask == objectID, dtype=np.uint8)
        img *= 255
        cv2.imwrite(path, img)

    def save_depth_image(self, path):
        img = self.depthImage.astype(np.float32)
        cv2.imwrite(path, img)
