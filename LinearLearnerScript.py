import pandas as pd
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.image_uris import retrieve
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer

# Load the housing CSV (replace this S3 URI with your own from the housing.csv files 
data = pd.read_csv("s3://amazon-sagemaker-438465137857-us-east-1-44ymcqbpjvkxpj/shared/housing.csv")
data.head()  # View the first few rows of the dataset (5 by default)

# Reorder the columns in the CSV to put house price first
TARGET = "median_house_value"
cols = [TARGET] + [col for col in data.columns if col != TARGET]
data = data[cols]

# Drop any non-numeric columns, since Linear Learner only handles numeric values
data = data.select_dtypes(include="number")
# Drop any rows with missing values
data = data.dropna()

# Drop the header and save the cleaned data!
data.to_csv("cleaned_housing.csv", index=False, header=False)

# Upload the data to the S3 that SageMaker made for us

# First, define some variables we'll use multiple times below:
session = sagemaker.Session()  # Establish the session
role = get_execution_role()  # Gets the IAM role SageMaker made
bucket = session.default_bucket()  # Gets the default S3 bucket SageMaker made
prefix = "linear-learner-housing"  # This will be the folder path in the bucket

# Upload the data to the default S3 bucket
s3_input_path = session.upload_data(
    path="cleaned_housing.csv",  # The cleaned data we made above
    bucket=bucket,  # Default S3 bucket SageMaker made for us
    key_prefix=prefix
)

print("Uploaded to: ", s3_input_path)

# Configure the Linear Learner Algorithm (SageMaker provides this!)
container = retrieve(
    framework="linear-learner",  # Grab SageMaker Linear Learner algorithm
    region=session.boto_region_name  # Grab the AWS region you're operating in
)

# Configure the estimator object, which will make our prediction
ll_estimator = Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=session
)

# Setting Hyperparameters for regression
# A Hyperparameter is just a param that defines how the model learns
ll_estimator.set_hyperparameters(
    predictor_type="regressor",
    mini_batch_size=32,  # Update weights after every 32 records
    epochs=10,  # 10 runs through the data
    feature_dim=data.shape[1] - 1  # minus target column, which we put first
)

# Start the training job! This launches the managed SageMaker job
train_input = TrainingInput(s3_input_path, content_type="text/csv")
ll_estimator.fit({"train": train_input})

# Deploy the model so we can use it
predictor = ll_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

# FINALLY, make the prediction!

# Tell SageMaker to format the input data as CSV
predictor.serializer = CSVSerializer()

# Get the first row of data without the target row.
# (0 - first row. 1 - skip target)
# Convert to float and reshape into a 2D array (Linear Learner needs it)
sample = data.iloc[0, 1:].values.astype(np.float32).reshape(1, -1)

# Get the Linear Learner to predict the price of the sample and print it out!
result = predictor.predict(sample)
print(result)

# IMPORTANT - delete the endpoint after running to avoid billing
predictor.delete_endpoint()
