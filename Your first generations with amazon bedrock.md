# Your First Generation With Amazon Bedrock
# Import necessary packages
import boto3
import json

# Setup Bedrock runtime
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

# Prompt Amazon Bedrock
prompt = "Write a one-sentence summary of Nairobi"

# Prepare keyword arguments for invoking the model
kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps({
        "inputText": prompt
    })
}

# Invoke the model and get the response
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

![image](https://github.com/charity-12/Serverless-LLM-apps-with-Amazon-Bedrock/assets/93730840/d9cfaff1-d2c1-40be-be1b-969e92ad4bc6)


# Print the formatted response
print(json.dumps(response_body, indent=4))

![image](https://github.com/charity-12/Serverless-LLM-apps-with-Amazon-Bedrock/assets/93730840/186c5278-d06d-4bec-8e12-9ad95c6aac89)


# Access and print the outputText from the results
output_text = response_body['results'][0]['outputText']
print(output_text)

![image](https://github.com/charity-12/Serverless-LLM-apps-with-Amazon-Bedrock/assets/93730840/8c86ee36-fac1-4399-9293-6f5373179413)


# Generation Configuration - First Attempt
prompt = "Write a summary of Mombasa."
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)
print(json.dumps(response_body, indent=4))

# Generation Configuration - Second Attempt
prompt = "Write a summary of Las Vegas."
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)
print(json.dumps(response_body, indent=4))

![image](https://github.com/charity-12/Serverless-LLM-apps-with-Amazon-Bedrock/assets/93730840/c0182849-164e-4c07-8dfb-b3dd1e71a080)


# Working with audio data type
from IPython.display import Audio, display

# Load and display the audio
audio = Audio(filename="dialog.mp3")
display(audio)

# Read transcript from a text file
transcript_file_path = 'transcript.txt'
with open(transcript_file_path, "r") as file:
    dialogue_text = file.read()

# Print the transcript
print(dialogue_text)

![image](https://github.com/charity-12/Serverless-LLM-apps-with-Amazon-Bedrock/assets/93730840/8b692f6d-57d5-4de2-b272-57c19a8c7fc8)

