
import os
import sys
import copy
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.NearestObject.src.utils.response import build_response
from components.NearestObject.src.models.PackageModel import PackageModel


class NearestObject(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.roi = json.loads(self.request.get_param("configRoi"))
        self.detections_input = self.request.get_param("inputDetections")
        self.measure_target = self.request.get_param("configMeasureType")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        roi_lines = self.roi.get('lines', [])
        if len(roi_lines) < 2:
            raise ValueError("ERROR: At least 2 lines (ROI) are required for measurement.")

        ref_line_1 = roi_lines[0]
        ref_line_2 = roi_lines[1]
        x_ref_left = (ref_line_1['x1'] + ref_line_1['x2']) / 2
        x_ref_right = (ref_line_2['x1'] + ref_line_2['x2']) / 2

        points_coords = []
        points_source = []

        if self.detections_input:
            for detection in self.detections_input:
                if self.measure_target == "keyPoints":
                    kps = detection.get('keyPoints', [])
                    if kps:
                        for kp in kps:
                            points_coords.append([kp['cx'], kp['cy']])
                            points_source.append(detection)

                elif self.measure_target == "boundingBox":
                    bbox = detection.get('boundingBox')
                    if bbox:
                        cx = bbox['x'] + (bbox['w'] / 2)
                        cy = bbox['y'] + (bbox['h'] / 2)
                        points_coords.append([cx, cy])
                        points_source.append(detection)

        if not points_coords:
            self.detections = []
            return build_response(context=self)

        coordinates_xy = np.array(points_coords)
        all_x_values = coordinates_xy[:, 0]
        dist_to_ref1 = np.abs(all_x_values - x_ref_left)
        idx_1 = np.argmin(dist_to_ref1)
        point_1 = coordinates_xy[idx_1]
        dist_to_ref2 = np.abs(all_x_values - x_ref_right)
        idx_2 = np.argmin(dist_to_ref2)
        point_2 = coordinates_xy[idx_2]
        euclidean_distance = np.linalg.norm(point_1 - point_2)
        source_detection = points_source[idx_1]
        result_detection = copy.deepcopy(source_detection)
        if 'attributes' not in result_detection:
            result_detection['attributes'] = {}
        result_detection['attributes']['distance_px'] = float(euclidean_distance)
        result_detection['attributes']['mode'] = self.measure_target
        result_detection['keyPoints'] = [
            {"cx": float(point_1[0]), "cy": float(point_1[1])},
            {"cx": float(point_2[0]), "cy": float(point_2[1])}
        ]
        self.detections = [result_detection]

        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()