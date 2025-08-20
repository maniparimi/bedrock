import boto3
import botocore.config
import json
import response

def blog_generate_using_bedrock(blog_topic: str) -> str:
    prompt = f"<s>[INST] <<SYS>>\nWrite a 200 words blog on the topic {blog_topic}\n<</SYS>>\n[/INST]"
    body = {
        "prompt": prompt,
        "max_gen_length": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=botocore.config.Config(
                read_timeout=300,
                retries={"max_attempts": 3}
            )
        )
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId="meta.llama2-13b-chat-v1"
        )
        response_content = response.get("body").read()
        response_data = json.loads(response_content)
        blog_details = response_data["generation"]
        return blog_details
    except Exception as e:
        print(f"Error generating the blog: {e}")
        return ""
    



from datetime import datetime

def lambda_handler(event, context):
    event = json.loads(event["body"])
    blog_topic = event["blog_topic"]
    generated_blog = blog_generate_using_bedrock(blog_topic=blog_topic)
    if generated_blog:
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"blog_output/{current_time}.txt"
        s3_bucket = "aws-bedrock-course-one"
        save_blog_details_in_s3(s3_key, s3_bucket, generated_blog)
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Blog generation is completed."})
        }
    else:
        print("No blog was generated.")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Blog generation failed."})
        }
    


def save_blog_details_in_s3(s3_key, s3_bucket, blog_content):
    s3 = boto3.client("s3")
    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=blog_content)
    except Exception as e:
        print(f"Error saving blog to S3: {e}")


