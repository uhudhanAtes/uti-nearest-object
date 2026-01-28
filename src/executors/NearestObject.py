import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.NearestObject.src.utils.response import build_response
from components.NearestObject.src.models.PackageModel import PackageModel


class NearestObject(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        print(self.detections)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        self.detections = []
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()