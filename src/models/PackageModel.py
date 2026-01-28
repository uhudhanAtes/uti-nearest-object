
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Detection


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection]]
    type: str = "list"

    class Config:
        title = "Detections"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: Union[List[Detection]]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class ConfigRoi(Config):
    """
        Configures the Region of Interest (ROI) for proximity detection.
        - This widget allows the user to draw reference lines on the image.
        - Users must draw exactly two lines to define the proximity zones.
        - The system will identify the object nearest to these drawn lines.
    """
    name: Literal["configRoi"] = "configRoi"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        json_schema_extra = {
            "shortDescription": "Reference Lines",
            "class": "\\novavision\\app\\widgets\\ROI",
            "options": {
                "isMultiple": "true",
                "name": "roi",
                "availableTypes": ["line"],
            },
        }
        title = "Roi"


class NearestObjectInputs(Inputs):
    inputDetections: InputDetections


class NearestObjectConfigs(Configs):
    configRoi: ConfigRoi


class NearestObjectOutputs(Outputs):
    outputDetections: OutputDetections


class NearestObjectRequest(Request):
    inputs: Optional[NearestObjectInputs]
    configs: NearestObjectConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class NearestObjectResponse(Response):
    outputs: NearestObjectOutputs


class NearestObject(Config):
    name: Literal["NearestObject"] = "NearestObject"
    value: Union[NearestObjectRequest, NearestObjectResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "NearestObject"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[NearestObject]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["NearestObject"] = "NearestObject"
