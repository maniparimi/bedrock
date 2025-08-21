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
import os

# Initialize AWS clients
ddb = boto3.resource('dynamodb', region_name=os.environ['AWS_REGION'])
bedrock = boto3.client('bedrock-runtime')

def lambda_handler(event, context):   
    resp_url = os.environ['ApiGatewayEndpoint']
    conntable = ddb.Table(os.environ['DDBTableName']) 
    apig_management_client = boto3.client(
        'apigatewaymanagementapi', 
        endpoint_url=resp_url
    )
    try:
        for record in event['Records']:
            request = json.loads(record["body"])
            name_value = request["name"]
            # Query DynamoDB for Connection
            response = conntable.query(
                IndexName='UsernameIndex',
                KeyConditionExpression='username = :username', #Task 2: Fill in this value
                ExpressionAttributeValues={':username': name_value}
            )
            connectionData = response['Items']
            
            if not connectionData:
                print(f"No connections found for username: {name_value}")
                continue

            prompt_data = request["prompt"]
            modelId = "us.amazon.nova-pro-v1:0"

            # Stream from Bedrock
            response = bedrock.converse_stream(
                #Task 3: Remove this comment and add the missing parameter in its place
                modelId=modelId,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt_data}]
                }],
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.5,
                    "topP": 0.9
                }
            )
            for event in response["stream"]:
                if 'contentBlockDelta' in event:
                    text = event["contentBlockDelta"]["delta"]["text"]
                    # Send to each connection
                    for conn in connectionData:
                        try:
                            send_response = apig_management_client.post_to_connection(
                                Data=json.dumps({"message": text}),  # Wrap in JSON
                                ConnectionId=conn['connectionId']
                            )
                        except apig_management_client.exceptions.GoneException:
                            print(f"Connection {conn['connectionId']} is gone")
                            continue
                        except Exception as e:
                            print(f"Error sending to connection {conn['connectionId']}: {str(e)}")
                            continue
    except Exception as e:
        print(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Messages sent successfully'})
    }
