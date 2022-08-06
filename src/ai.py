# Credits: https://github.com/QIN2DIM/hcaptcha-challenger

from solutions import sk_recognition, resnet, yolo


class AI:
    label_alias = {
        "airplane": "airplane",
        "motorbus": "bus",
        "bus": "bus",
        "truck": "truck",
        "motorcycle": "motorcycle",
        "boat": "boat",
        "bicycle": "bicycle",
        "train": "train",
        "vertical river": "vertical river",
        "airplane in the sky flying left": "airplane in the sky flying left",
        "Please select all airplanes in the sky that are flying to the right": "airplanes in the sky that are flying to the right",
        "car": "car",
        "elephant": "elephant",
        "parrot": "bird",
        "bird": "bird",
    }

    def __init__(
            self,
            dir_model: str = None,
            onnx_prefix: str = None,
            path_objects_yaml: str = None,
            path_rainbow_yaml: str = None,
    ):

        self.dir_model = dir_model
        self.onnx_prefix = onnx_prefix
        self.path_objects_yaml = path_objects_yaml
        self.path_rainbow_yaml = path_rainbow_yaml

        self.pom_handler = resnet.PluggableONNXModels(self.path_objects_yaml)
        self.label_alias.update(self.pom_handler.label_alias["en"])
        self.pluggable_onnx_models = self.pom_handler.overload(
            self.dir_model, path_rainbow=self.path_rainbow_yaml
        )
        self.yolo_model = yolo.YOLO(self.dir_model, self.onnx_prefix)

    def switch_solution(self, label):
        """Optimizing solutions based on different challenge labels"""
        sk_solution = {
            "vertical river": sk_recognition.VerticalRiverRecognition,
            "airplane in the sky flying left": sk_recognition.LeftPlaneRecognition,
            "airplanes in the sky that are flying to the right": sk_recognition.RightPlaneRecognition,
        }

        label_alias = self.label_alias.get(label)

        # Select ResNet ONNX model
        if self.pluggable_onnx_models.get(label_alias):
            return self.pluggable_onnx_models[label_alias]
        # Select SK-Image method
        if sk_solution.get(label_alias):
            return sk_solution[label_alias](self.path_rainbow_yaml)
        # Select YOLO ONNX model
        return self.yolo_model

    def predict(self, image_bytes, label):
        model = self.switch_solution(label)
        return model.solution(img_stream=image_bytes, label=self.label_alias[label])
