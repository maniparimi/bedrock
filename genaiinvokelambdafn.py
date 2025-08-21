import json
import boto3

# Initialize client libraries
bedrock = boto3.client("bedrock-runtime")

# Initialize variable for Bedrock
NOVA_MODEL_ID = "us.amazon.nova-lite-v1:0"
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