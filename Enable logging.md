# Enable Logging for Amazon Bedrock LLM Calls

## Introduction

This document provides a step-by-step guide on how to enable logging for all calls made to and responses received from the Large Language Models (LLM) within Amazon Bedrock. Logging is crucial for monitoring and troubleshooting purposes. We will leverage Amazon CloudWatch for logging and set up the necessary configurations.

## Prerequisites

Make sure you have the following prerequisites in place before proceeding:

- AWS Account
- Appropriate IAM permissions to configure CloudWatch and S3
- Python environment with necessary dependencies (boto3, helpers module)

## Implementation

```python
# Import Necessary Libraries
import boto3
import json
import os

# Initialize Bedrock Client
bedrock = boto3.client('bedrock', region_name="us-west-2")

# Set up CloudWatch
from helpers.CloudWatchHelper import CloudWatch_Helper
cloudwatch = CloudWatch_Helper()
log_group_name = '/my/amazon/bedrock/logs'
cloudwatch.create_log_group(log_group_name)

# Configure CloudWatch Helper Function
loggingConfig = {
    'cloudWatchConfig': {
        'logGroupName': log_group_name,
        'roleArn': os.environ['LOGGINGROLEARN'],
        'largeDataDeliveryS3Config': {
            'bucketName': os.environ['LOGGINGBUCKETNAME'],
            'keyPrefix': 'amazon_bedrock_large_data_delivery',
        }
    },
    's3Config': {
        'bucketName': os.environ['LOGGINGBUCKETNAME'],
        'keyPrefix': 'amazon_bedrock_logs',
    },
    'textDataDeliveryEnabled': True,
}

# Configure Bedrock for Model Invocation Logging
bedrock.put_model_invocation_logging_configuration(loggingConfig=loggingConfig)

# Initialize Bedrock Runtime Client
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-west-2")

# Sample Model Invocation
prompt = "Write an article about the fictional planet Foobar."

kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

# Invoke Model
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

# Extract Generated Text
generation = response_body['results'][0]['outputText']

# Print Generated Text
print(generation)

# Access Recent Logs in CloudWatch
cloudwatch.print_recent_logs(log_group_name)

#AWS Console Access
from IPython.display import HTML

# Get AWS Console URL from Environment Variables
aws_url = os.environ['AWS_CONSOLE_URL']

# Display AWS Console Link
HTML(f'<a href="{aws_url}" target="_blank">GO TO AWS CONSOLE</a>')

