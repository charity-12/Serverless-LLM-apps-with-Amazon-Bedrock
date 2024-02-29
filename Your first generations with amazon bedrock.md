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

# Print the formatted response
print(json.dumps(response_body, indent=4))

# Access and print the outputText from the results
output_text = response_body['results'][0]['outputText']
print(output_text)

# Generation Configuration - First Attempt
prompt = "Write a summary of Las Vegas."
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

Working with audio data type
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
