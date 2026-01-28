
from sdks.novavision.src.helper.package import PackageHelper
from components.NearestObject.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, NearestObjectOutputs, NearestObjectResponse, NearestObject, OutputDetections


def build_response(context):
    outputImage = OutputDetections(value=context.image)
    Outputs = NearestObjectOutputs(outputImage=outputImage)
    packageResponse = NearestObjectResponse(outputs=Outputs)
    packageExecutor = NearestObject(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel