import argparse
import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp import components


train_r_component_file = 'image_code/00_r_train/component.yaml'
deploy_r_component_file = 'image_code/01_r_deploy/component.yaml'


# Components load
train_r_component = components.load_component_from_file(train_r_component_file)
deploy_r_component = components.load_component_from_file(deploy_r_component_file)


def main(params):
    print('Generating and executing R train-deploy pipeline ...')
    model_file_name = params.model_file_name
    gcp_bucket=params.gcp_bucket
    namespace = params.namespace

    @dsl.pipeline(
        name='R train-deploy pipelines',
        description='Sample kubeflow pipeline with R / xgboost'
    )
    def r_train_deploy_pipeline(
    ):
        r_train_task = train_r_component(model_file_name, gcp_bucket)
        r_deploy_task = deploy_r_component(namespace,model_file_name, gcp_bucket).after(r_train_task)




    # Generate .zip file
    pipeline_func = r_train_deploy_pipeline
    pipeline_filename = pipeline_func.__name__ + '.zip'
    compiler.Compiler().compile(pipeline_func, pipeline_filename)


    # Define the client
    host = params.host
    client_id = params.client_id
    other_client_id = params.other_client_id
    other_client_secret = params.other_client_secret
    namespace = params.namespace

    client = kfp.Client(host=host, client_id=client_id, namespace=namespace, other_client_id=other_client_id,
                        other_client_secret=other_client_secret)
    # Experiment
    experiment = client.create_experiment('r-pipeline-xgboost')
    experiment_name = 'r-pipeline-xgboost'
    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path=pipeline_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='R train-deploy kubeflow pipelines')
    parser.add_argument('--host', type=str)
    parser.add_argument('--client_id', type=str)
    parser.add_argument('--other_client_id', type=str)
    parser.add_argument('--other_client_secret', type=str)
    parser.add_argument('--namespace', type=str)
    parser.add_argument('--model_file_name', type=str)
    parser.add_argument('--gcp_bucket', type=str)

    params = parser.parse_args()
    main(params)

