import xml.etree.ElementTree as ET


class URDF:

    def __init__(self, pathToDummyUrdf="urdfs/objects/dummy_urdf.urdf"):
        self.root = self.load_urdf(pathToDummyUrdf)

    def load_urdf(self, pathToUrdf):
        tree = ET.parse(pathToUrdf)
        root = tree.getroot()
        return root

    def set_name(self, name):
        self.root.attrib["name"] = name

    def set_visual_mesh_path(self, pathToMesh):
        field = self.root.find(".//visual/geometry/mesh")
        field.attrib.update({"filename": pathToMesh})

    def set_visual_geometry_scale(self, scaleFactor):
        scale = f"{scaleFactor} {scaleFactor} {scaleFactor}"
        field = self.root.find(".//visual/geometry/mesh")
        field.attrib.update({"scale": scale})

    def set_collision_mesh_path(self, pathToMesh):
        field = self.root.find(".//collision/geometry/mesh")
        field.attrib.update({"filename": pathToMesh})

    def set_collision_geometry_scale(self, scaleFactor):
        scale = f"{scaleFactor} {scaleFactor} {scaleFactor}"
        field = self.root.find(".//collision/geometry/mesh")
        field.attrib.update({"scale": scale})

    def set_mass(self, massValue):
        mass = str(massValue)
        field = self.root.find(".//inertial/mass")
        field.attrib.update({"value": mass})

    def set_inertia(self, inertiaDict):
        ixx = str(inertiaDict["ixx"])
        ixy = str(inertiaDict["ixy"])
        ixz = str(inertiaDict["ixz"])
        iyy = str(inertiaDict["iyy"])
        iyz = str(inertiaDict["iyz"])
        izz = str(inertiaDict["izz"])
        field = self.root.find(".//inertial/inertia")
        field.attrib.update({"ixx": ixx, "ixy": ixy, "ixz": ixz, "iyy": iyy, "iyz": iyz, "izz": izz})

    def save(self, pathToUrdf):
        data = ET.tostring(self.root)
        with open(pathToUrdf, "wb") as f:
            f.write(data)
