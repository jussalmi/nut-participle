#Originally run as notebook in Microsoft Azure Machine Learning

import logging

from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig

train=pd.read_csv('Esuka_kierros2-train.csv', sep=";")
test=pd.read_csv('Esuka_kierros2-test.csv', sep=";")
NotPreds = ['target', 'ID', 'Osuma', 'Basic-Compound']
preds = [col for col in train.columns if col not in NotPreds]
print(preds)
ws = Workspace.from_config()

# choose a name for experiment
experiment_name = "automl-classification-ccard-remote"

experiment = Experiment(ws, experiment_name)

output = {}
output["Subscription ID"] = ws.subscription_id
output["Workspace"] = ws.name
output["Resource Group"] = ws.resource_group
output["Location"] = ws.location
output["Experiment Name"] = experiment.name
output["SDK Version"] = azureml.core.VERSION
pd.set_option("display.max_colwidth", None)
outputDf = pd.DataFrame(data=output, index=[""])
outputDf.T

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster-1"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS12_V2", max_nodes=6
    )
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True)


from azureml.core import Dataset
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import TabularDatasetFactory
import pandas as pd

# Convert the Pandas DataFrame to a TabularDataset
dataset_train = TabularDatasetFactory.register_pandas_dataframe(
    dataframe=train[preds+['target']],
    target=ws.get_default_datastore(),
    name="Train Dataset"
)
dataset_test = TabularDatasetFactory.register_pandas_dataframe(
    dataframe=test[preds+['target']],
    target=ws.get_default_datastore(),
    name="Test Dataset"
)


automl_settings = {
    "primary_metric": "average_precision_score_weighted",
    "enable_early_stopping": True,
    "max_concurrent_iterations": 2,  # This is a limit for testing purpose, please increase it as per cluster size
    "experiment_timeout_hours": 0.25,  # This is a time limit for testing purposes, remove it for real use cases, this will drastically limit ablity to find the best model possible
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(
    task="classification",
    debug_log="automl_errors.log",
    compute_target=compute_target,
    training_data=dataset_train,
    test_data=dataset_test,
    label_column_name="target",
    **automl_settings,
)

remote_run = experiment.submit(automl_config, show_output=True)

experiment.get_portal_url()
