class CameraSettings:
    def __init__(self, ret=None, camera_matrix=None, distortion_coefficients=None, rotation_vector=None, translation_vector=None):
        self.ret = ret
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector


cameraSettings = CameraSettings()
