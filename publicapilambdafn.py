"""
MIT No Attribution
Copyright 2025 Amazon

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json
import boto3

# Initialize client libraries
bedrock = boto3.client("bedrock-runtime")

# Initialize variable for Bedrock
NOVA_MODEL_ID = "us.amazon.nova-micro-v1:0"
SYSTEM_MESSAGE = "You are a a helpful travel booking assistant."

def lambda_handler(event, context):
    """Lambda handler - invokes bedrock with the input coming from event object"""
    bedrock_model_response = invoke_bedrock(event["prompt"])
    return bedrock_model_response
    
def invoke_bedrock(user_request):
    """
    Function to invoke Amazon Nova using the Converse API
    Creates a request with messages and inference configuration
    Returns the response text from the model
    """
    system_prompts = [{"text": SYSTEM_MESSAGE}]
    messages = [
        {
            "role": "user",
            "content": [{"text": user_request}]
        }
    ]

    # Setup inference configuration
    inference_config = {
        "temperature": 0,
        "maxTokens": 2048,
        "topP": 0.9
    }

    # Send the request to the model
    response = bedrock.converse(
        modelId=NOVA_MODEL_ID,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config
    )
    response_body = response.get("output")
    return response_body["message"]["content"][0]["text"]
