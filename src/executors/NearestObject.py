
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
        if not roi_lines:
            raise ValueError("ERROR: At least 1 line (ROI) is required.")

        points_coords = []
        points_source = []
        if self.detections_input:
            for detection in self.detections_input:
                if self.measure_target == "keyPoints":
                    kps = detection.get('keyPoints', [])
                    for kp in kps:
                        points_coords.append([kp['cx'], kp['cy']])
                        points_source.append(detection)
                elif self.measure_target == "boundingBox":
                    bbox = detection.get('boundingBox')
                    if bbox:
                        cx = bbox['left'] + (bbox['width'] / 2)
                        cy = bbox['top'] + (bbox['height'] / 2)
                        points_coords.append([cx, cy])
                        points_source.append(detection)

        if not points_coords:
            self.detections = []
            return build_response(context=self)

        coordinates_xy = np.array(points_coords)
        if self.measure_target == "keyPoints":
            new_detections = []

            for idx, line in enumerate(roi_lines):
                l_start = [line['x1'], line['y1']]
                l_end = [line['x2'], line['y2']]
                distances = self.get_distance_point_to_segment(coordinates_xy, l_start, l_end)
                nearest_idx = np.argmin(distances)
                new_obj = copy.deepcopy(points_source[nearest_idx])
                new_obj['classLabel'] = f"k{idx + 1}"
                new_obj['keyPoints'] = [{
                    "cx": float(coordinates_xy[nearest_idx][0]),
                    "cy": float(coordinates_xy[nearest_idx][1])
                }]

                new_detections.append(new_obj)
            self.detections = new_detections

        else:
            selected_detections = []
            selected_indices = set()

            for line in roi_lines:
                l_start = [line['x1'], line['y1']]
                l_end = [line['x2'], line['y2']]
                distances = self.get_distance_point_to_segment(coordinates_xy, l_start, l_end)
                nearest_idx = np.argmin(distances)

                if nearest_idx not in selected_indices:
                    selected_detections.append(copy.deepcopy(points_source[nearest_idx]))
                    selected_indices.add(nearest_idx)

            self.detections = selected_detections

        print(self.detections)
        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()