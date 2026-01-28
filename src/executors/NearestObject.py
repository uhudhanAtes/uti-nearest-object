
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

    def get_distance_point_to_segment(self, points, line_start, line_end):
        """
            Numpy kullanarak çoklu noktaların bir doğru parçasına (segment) olan uzaklığını hesaplar.
            points: (N, 2) array [[x,y], [x,y]]
            line_start: [x, y]
            line_end: [x, y]
        """
        P = points
        A = np.array(line_start)
        B = np.array(line_end)
        AB = B - A
        AP = P - A
        len_AB_sq = np.dot(AB, AB)

        if len_AB_sq == 0:
            return np.linalg.norm(P - A, axis=1)
        t = np.sum(AP * AB, axis=1) / len_AB_sq
        t = np.clip(t, 0, 1)
        closest_points_on_line = A + t[:, np.newaxis] * AB
        distances = np.linalg.norm(P - closest_points_on_line, axis=1)

        return distances

    def run(self):
        roi_lines = self.roi.get('lines', [])
        if len(roi_lines) < 2:
            raise ValueError("ERROR: At least 2 lines (ROI) are required for measurement.")

        line1_start = [roi_lines[0]['x1'], roi_lines[0]['y1']]
        line1_end = [roi_lines[0]['x2'], roi_lines[0]['y2']]
        line2_start = [roi_lines[1]['x1'], roi_lines[1]['y1']]
        line2_end = [roi_lines[1]['x2'], roi_lines[1]['y2']]
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
        dist_to_line1 = self.get_distance_point_to_segment(coordinates_xy, line1_start, line1_end)
        idx_1 = np.argmin(dist_to_line1)
        point_1 = coordinates_xy[idx_1]
        dist_to_line2 = self.get_distance_point_to_segment(coordinates_xy, line2_start, line2_end)
        idx_2 = np.argmin(dist_to_line2)
        if idx_1 == idx_2 and len(coordinates_xy) > 1:
            dist_to_line2[idx_1] = np.inf
            idx_2 = np.argmin(dist_to_line2)

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