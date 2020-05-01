import logging

from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from sklearn.metrics import mean_squared_error, r2_score

ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'automl-airbnb'

experiment=Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

# Choose a name for your AmlCompute cluster.
amlcompute_cluster_name = "cpu-cluster-1"

found = False
# Check if this compute target already exists in the workspace.
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'cpu-cluster-1':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]
    
if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_DS12_V2", # for GPU, use "STANDARD_NC6"
                                                                #vm_priority = 'lowpriority', # optional
                                                                max_nodes = 6)

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)
    
print('Checking cluster status...')
# Can poll for a minimum number of nodes and for a specific timeout.
# If no min_node_count is provided, it will use the scale settings for the cluster.
compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)


data = 'https://query.data.world/s/nmz3slczc3c4orpadnuxrqvw54uxd4'
dataset = Dataset.Tabular.from_delimited_files('https://query.data.world/s/nmz3slczc3c4orpadnuxrqvw54uxd4')


dataset = dataset.drop_columns('room_id')
dataset = dataset.drop_columns('survey_id')
dataset = dataset.drop_columns('host_id')
dataset = dataset.drop_columns('latitude')
dataset = dataset.drop_columns('longitude')
dataset = dataset.drop_columns('name')
dataset = dataset.drop_columns('Column1')
dataset = dataset.drop_columns('last_modified')
dataset.take(3).to_pandas_dataframe()
train_data, test_data = dataset.random_split(percentage=0.8, seed=223)
label = "price"

automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": 'r2_score',
    "preprocess": True,
    "enable_early_stopping": True, 
    "experiment_timeout_minutes": 20, #for real scenarios we reccommend a timeout of at least one hour 
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'regression',
                             compute_target = compute_target,
                             training_data = train_data,
                             label_column_name = label,
                             **automl_settings
                            )

remote_run = experiment.submit(automl_config, show_output = False)

from azureml.widgets import RunDetails
RunDetails(remote_run).show()

remote_run.wait_for_completion(show_output=False)

best_run, fitted_model = remote_run.get_output()
print(best_run)
print(fitted_model)

# preview the first 3 rows of the dataset

test_data = test_data.to_pandas_dataframe()
y_test = test_data[label].fillna(0)
test_data = test_data.drop(label, 1)
test_data = test_data.fillna(0)


train_data = train_data.to_pandas_dataframe()
y_train = train_data[label].fillna(0)
train_data = train_data.drop(label, 1)
train_data = train_data.fillna(0)

y_pred_train = fitted_model.predict(train_data)
y_residual_train = y_train - y_pred_train

y_pred_test = fitted_model.predict(test_data)
y_residual_test = y_test - y_pred_test

test_pred = plt.scatter(y_test, y_pred_test, color='r')
test_test = plt.scatter(y_test, y_test, color='g')
plt.legend((test_pred, test_test), ('prediction', 'truth'), loc='upper left', fontsize=8)
plt.show()

