import time

import cv2
import numpy as np

from point import Point3D, Point2D


class Rectangle:

    def __init__(self, p1, p2, p3, p4):
        assert isinstance(p1, Point2D) or isinstance(p1, Point3D), "p1 must be a 2D or 3D Point!"
        assert isinstance(p2, Point2D) or isinstance(p2, Point3D), "p2 must be a 2D or 3D Point!"
        assert isinstance(p3, Point2D) or isinstance(p3, Point3D), "p3 must be a 2D or 3D Point!"
        assert isinstance(p4, Point2D) or isinstance(p4, Point3D), "p4 must be a 2D or 3D Point!"

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def __iter__(self):
        yield self.p1
        yield self.p2
        yield self.p3
        yield self.p4

    @property
    def dimension(self):
        list2D = []
        list3D = []
        for point in self:
            list2D.append(isinstance(point, Point2D))
            list3D.append(isinstance(point, Point3D))
        if all(list2D):
            return "2D"
        elif all(list3D):
            return "3D"
        else:
            raise TypeError("All points must have the same dimension, either 2D or 3D!")

    @property
    def center(self):
        if self.dimension == "2D":
            x, y = 0, 0
            for point in self:
                x += point.x
                y += point.y
            x /= 4
            y /= 4
            center = Point2D(x, y)
        elif self.dimension == "3D":
            x, y, z = 0, 0, 0
            for point in self:
                x += point.x
                y += point.y
                z += point.z
            x /= 4
            y /= 4
            z /= 4
            center = Point3D(x, y, z)
        else:
            raise NotImplemented
        return center

    @property
    def width(self):
        w = self.p2.distance_to(self.p3)
        return w

    @property
    def height(self):
        h = self.p1.distance_to(self.p2)
        return h

    @property
    def theta(self):
        if self.p2.x == self.p1.x:
            if self.p2.y > self.p1.y:
                theta = np.pi / 2
            elif self.p2.y < self.p1.y:
                theta = -np.pi / 2
        else:
            theta = np.arctan((self.p2.y - self.p1.y) / (self.p2.x - self.p1.x))
        return theta

    def to_grasping_rectangle(self):
        return GraspingRectangle(self.center, self.width, self.height, self.theta)

    def rotate_in_gui(self, theta):
        rotationMatrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]])
        rotatedRectangle = []
        for point in self:
            rotatedPoint = np.matmul(rotationMatrix,
                                     point.to_np_array() - self.center.to_np_array()) + self.center.to_np_array()
            rotatedRectangle.append(rotatedPoint)

        for i, key in enumerate(self.__dict__):
            self.__dict__[key] = Point3D.from_np_array(rotatedRectangle[i])
            if i > 3:
                break

    def rotate_in_image(self, theta):
        rotationMatrix = np.array([[np.cos(theta), np.sin(theta)],
                                   [-np.sin(theta), np.cos(theta)]])

        rotatedRectangle = []
        for point in self:
            rotatedPoint = np.matmul(rotationMatrix,
                                     point.to_np_array() - self.center.to_np_array()) + self.center.to_np_array()
            rotatedRectangle.append(rotatedPoint)

        for i, key in enumerate(self.__dict__):
            self.__dict__[key] = Point2D.from_np_array(rotatedRectangle[i])
            if i > 3:
                break

    def draw_in_gui(self, bulletClient, displayDuration=0):
        bulletClient.addUserDebugLine(self.p1.to_list(), self.p2.to_list(), [0, 0, 0], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p2.to_list(), self.p3.to_list(), [0, 0, 0], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p3.to_list(), self.p4.to_list(), [0, 0, 0], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p4.to_list(), self.p1.to_list(), [0, 0, 0], 2, lifeTime=displayDuration)
        time.sleep(1)

    def draw_in_image(self, inputImagePath, outputImagePath):
        img = cv2.imread(inputImagePath)
        cv2.line(img, self.p1.to_int_tuple(), self.p2.to_int_tuple(), (0, 0, 0), 2)
        cv2.line(img, self.p2.to_int_tuple(), self.p3.to_int_tuple(), (0, 0, 0), 2)
        cv2.line(img, self.p3.to_int_tuple(), self.p4.to_int_tuple(), (0, 0, 0), 2)
        cv2.line(img, self.p4.to_int_tuple(), self.p1.to_int_tuple(), (0, 0, 0), 2)
        cv2.imwrite(outputImagePath, img)


class AARectangle(Rectangle):

    def __init__(self, topLeft, topRight, bottomRight, bottomLeft):
        super().__init__(topLeft, topRight, bottomRight, bottomLeft)

    @property
    def topLeft(self):
        return self.p1

    @property
    def topRight(self):
        return self.p2

    @property
    def bottomRight(self):
        return self.p3

    @property
    def bottomLeft(self):
        return self.p4

    def enlarge(self, d):
        self.bottomRight.x = self.bottomRight.x + d
        self.bottomRight.y = self.bottomRight.y - d
        self.bottomLeft.x = self.bottomLeft.x - d
        self.bottomLeft.y = self.bottomLeft.y - d
        self.topLeft.x = self.topLeft.x - d
        self.topLeft.y = self.topLeft.y + d
        self.topRight.x = self.topRight.x + d
        self.topRight.y = self.topRight.y + d

    def contains(self, object):
        if isinstance(object, Point2D) or isinstance(object, Point3D):
            point = object
            isInside = (self.bottomLeft.x <= point.x <= self.bottomRight.x) and \
                       (self.bottomLeft.y <= point.y <= self.topLeft.y)
        elif isinstance(object, AARectangle):
            rectangle = object
            isInside = True
            for point in rectangle:
                if not self.contains(point):
                    return False
        else:
            raise ValueError("Input must be a point or a rectangle!")
        return isInside

    def is_outside(self, rectangle):
        assert isinstance(rectangle, AARectangle), "Input must be an axis aligned rectangle!"
        for point in self:
            if rectangle.contains(point):
                return False
        return True


class GraspingRectangle(Rectangle):

    def __init__(self, center, w, h, theta, graspSuccessful):
        p1, p2, p3, p4 = self.get_vertices(center, w, h, theta)
        super().__init__(p1, p2, p3, p4)
        self.graspSuccessful = graspSuccessful

    @staticmethod
    def get_vertices(center, w, h, theta):
        if isinstance(center, Point2D):
            p1 = Point2D(center.x - (w / 2) * np.cos(theta) - (h / 2) * np.sin(theta),
                         center.y - (w / 2) * (-np.sin(theta)) - (h / 2) * np.cos(theta))
            p2 = Point2D(center.x - (w / 2) * np.cos(theta) + (h / 2) * np.sin(theta),
                         center.y - (w / 2) * (-np.sin(theta)) + (h / 2) * np.cos(theta))
            p3 = Point2D(center.x + (w / 2) * np.cos(theta) + (h / 2) * np.sin(theta),
                         center.y + (w / 2) * (-np.sin(theta)) + (h / 2) * np.cos(theta))
            p4 = Point2D(center.x + (w / 2) * np.cos(theta) - (h / 2) * np.sin(theta),
                         center.y + (w / 2) * (-np.sin(theta)) - (h / 2) * np.cos(theta))
            return p1, p2, p3, p4
        else:
            raise NotImplemented

    def draw_in_gui(self, bulletClient, displayDuration=0):
        bulletClient.addUserDebugLine(self.p1.to_list(), self.p2.to_list(), [0, 0, 1], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p2.to_list(), self.p3.to_list(), [1, 0, 0], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p3.to_list(), self.p4.to_list(), [0, 0, 1], 2, lifeTime=displayDuration)
        bulletClient.addUserDebugLine(self.p4.to_list(), self.p1.to_list(), [1, 0, 0], 2, lifeTime=displayDuration)
        time.sleep(1)

    def draw_in_image(self, inputImagePath, outputImagePath):
        img = cv2.imread(inputImagePath)
        # img = self.rgbImage.astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.line(img, self.p1.to_int_tuple(), self.p2.to_int_tuple(), (255, 0, 0), 2)
        if self.graspSuccessful:
            cv2.line(img, self.p2.to_int_tuple(), self.p3.to_int_tuple(), (0, 255, 0), 2)
        else:
            cv2.line(img, self.p2.to_int_tuple(), self.p3.to_int_tuple(), (0, 0, 255), 2)
        cv2.line(img, self.p3.to_int_tuple(), self.p4.to_int_tuple(), (255, 0, 0), 2)
        if self.graspSuccessful:
            cv2.line(img, self.p4.to_int_tuple(), self.p1.to_int_tuple(), (0, 255, 0), 2)
        else:
            cv2.line(img, self.p4.to_int_tuple(), self.p1.to_int_tuple(), (0, 0, 255), 2)
        cv2.imwrite(outputImagePath, img)
