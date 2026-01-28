
from sdks.novavision.src.helper.package import PackageHelper
from components.NearestObject.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, NearestObjectOutputs, NearestObjectResponse, NearestObject, OutputDetections


def build_response(context):
    outputDetections = OutputDetections(value=context.detections)
    Outputs = NearestObjectOutputs(outputDetections=outputDetections)
    packageResponse = NearestObjectResponse(outputs=Outputs)
    packageExecutor = NearestObject(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel