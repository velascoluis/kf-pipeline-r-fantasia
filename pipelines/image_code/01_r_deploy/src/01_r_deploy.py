import argparse
import datetime
from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2XGBoostSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements


def deploy_model(namespace,model_file_name,gcp_bucket):


    api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    inference_service_name = 'xgboost-r'+now
    default_endpoint_spec = V1alpha2EndpointSpec(
        predictor=V1alpha2PredictorSpec(
            min_replicas=1,
            xgboost=V1alpha2XGBoostSpec(
        #storage_uri='gs://'+gcp_bucket+'/rmodel/'+model_file_name,
         storage_uri='gs://'+gcp_bucket+'/rmodel',
        resources=V1ResourceRequirements(
        requests={'cpu': '100m', 'memory': '1Gi'},
        limits={'cpu': '100m', 'memory': '1Gi'}))))

    isvc = V1alpha2InferenceService(api_version=api_version,
                                    kind=constants.KFSERVING_KIND,
                                    metadata=client.V1ObjectMeta(
                                                name=inference_service_name,
                                                namespace=namespace,
                                                annotations={'sidecar.istio.io/inject': 'false'}),
                                    spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))

    #@velascoluis - annotation The sidecar.istio.io/inject: "false", otherwise the ingress does not work

    KFServing = KFServingClient()
    KFServing.create(isvc)
    KFServing.get(inference_service_name, namespace=namespace, watch=True, timeout_seconds=120)


def main(params):
    deploy_model(params.namespace, params.model_file_name, params.gcp_bucket)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy xgboost model')
    parser.add_argument('--namespace',       type=str, default='default')
    parser.add_argument('--model_file_name', type=str, default='default')
    parser.add_argument('--gcp_bucket',      type=str, default='default')
    params = parser.parse_args()
    main(params)