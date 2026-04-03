# ============================================================================
# storage_handler.py
# Handles all AWS storage operations for the medical chatbot.
# Two storage targets:
#   1. S3  — stores raw user input files (audio, images, text)
#             and final chat JSON output as backup log
#   2. DynamoDB — stores the full chat record as the primary database
# ============================================================================

import os
import json
import copy
import boto3
from botocore.exceptions import ClientError

# ============================================================================
# Read AWS credentials from environment variables.
# On Modal these come from the "aws-secret" Modal secret.
# Locally they come from a .env file or shell environment.
# ============================================================================
AWS_ACCESS_KEY    = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY    = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_REGION         = os.getenv("AWS_REGION_S3", "us-east-1")
S3_BUCKET         = os.getenv("S3_BUCKET_NAME", "")
DYNAMODB_REGION   = os.getenv("AWS_REGION_DYNAMODB", "us-east-2")
DYNAMODB_TABLE    = os.getenv("DYNAMODB_TABLE_NAME", "")


def _get_s3_client():
    """Creates and returns a boto3 S3 client using env credentials."""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION
    )


def _get_dynamodb_resource():
    """Creates and returns a boto3 DynamoDB resource using env credentials."""
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=DYNAMODB_REGION
    )


# ============================================================================
# upload_input_to_s3()
# Uploads raw user input (text/audio/image) to S3.
# S3 path structure:
#   {user_id}/{chat_id}/chats/{query_id}/input/{filename}
# Example:
#   user123/chat_001/chats/1/input/audio.wav
#   user123/chat_001/chats/1/input/image_1.jpg
# Returns the permanent s3:// URI string.
# ============================================================================
def upload_input_to_s3(
    file_bytes: bytes,
    user_id: str,
    chat_id: str,
    query_id: int,
    input_type: str,
    file_extension: str
) -> str:
    s3 = _get_s3_client()

    if input_type == "images":
        file_name = f"image_1.{file_extension}"
    elif input_type == "audio":
        file_name = f"audio.{file_extension}"
    else:
        file_name = f"query_text.{file_extension}"

    object_key = f"{user_id}/{chat_id}/chats/{query_id}/input/{file_name}"
    s3_uri = f"s3://{S3_BUCKET}/{object_key}"

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=object_key, Body=file_bytes)
        print(f"S3 upload OK: {s3_uri}")
        return s3_uri
    except ClientError as e:
        print(f"S3 upload error: {e}")
        return ""


# ============================================================================
# generate_presigned_url()
# Converts a private s3:// URI into a temporary HTTPS URL.
# The URL expires after expiry_seconds (default 1 hour).
# Used so the AI model can securely access private user files
# without making the S3 bucket public.
# ============================================================================
def generate_presigned_url(s3_uri: str, expiry_seconds: int = 3600) -> str:
    if not s3_uri.startswith(f"s3://{S3_BUCKET}/"):
        print(f"Invalid S3 URI: {s3_uri}")
        return ""

    object_key = s3_uri.replace(f"s3://{S3_BUCKET}/", "")
    s3 = _get_s3_client()

    try:
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': S3_BUCKET, 'Key': object_key},
            ExpiresIn=expiry_seconds
        )
        return url
    except ClientError as e:
        print(f"Presigned URL error: {e}")
        return ""


def _sync_query_num(record: dict) -> None:
    """
    Auto-calculates query_num list from the chats dictionary.
    Keeps query_num in sync without manual counting.
    Example: chats has query_1, query_2 → query_num = [1, 2]
    """
    if "chats" in record and isinstance(record["chats"], dict):
        nums = []
        for k in record["chats"].keys():
            if k.startswith("query_"):
                try:
                    nums.append(int(k.split("_")[1]))
                except ValueError:
                    pass
        if nums:
            record["query_num"] = sorted(nums)


# ============================================================================
# store_output_to_dynamodb()
# Saves the full chat record to DynamoDB.
# PK = user_id, SK = chat_id
# Safe overwriting: if the same user sends a 2nd message, it updates
# the existing record instead of creating a duplicate.
# Auto-sync: automatically calculates query_num from chats keys.
# ============================================================================
def store_output_to_dynamodb(record: dict) -> None:
    dynamodb = _get_dynamodb_resource()
    table = dynamodb.Table(DYNAMODB_TABLE)

    _sync_query_num(record)

    try:
        table.put_item(Item=record)
        print(f"DynamoDB stored: user={record.get('user_id')} chat={record.get('chat_id')}")
    except ClientError as e:
        print(f"DynamoDB error: {e}")


# ============================================================================
# store_output_log_to_s3()
# Saves the full chat JSON as a backup log file to S3.
# S3 path: {user_id}/{chat_id}/output/chat.json
#
# Key difference from DynamoDB:
#   DynamoDB wants chats as a DICT  → {"query_1": {...}, "query_2": {...}}
#   S3 wants chats as a LIST ARRAY  → [{...}, {...}]
# This function auto-converts the format — you pass the same dict to both.
# ============================================================================
def store_output_log_to_s3(record: dict) -> str:
    user_id = record.get("user_id", "unknown")
    chat_id = record.get("chat_id", "unknown")
    object_key = f"{user_id}/{chat_id}/output/chat.json"
    s3_uri = f"s3://{S3_BUCKET}/{object_key}"

    s3 = _get_s3_client()
    try:
        _sync_query_num(record)

        # Deep copy so we don't mutate the original DynamoDB-format dict
        s3_record = copy.deepcopy(record)

        # Convert chats dict → list for S3 format
        if "chats" in s3_record and isinstance(s3_record["chats"], dict):
            s3_record["chats"] = list(s3_record["chats"].values())

        json_bytes = json.dumps(s3_record, indent=2).encode('utf-8')
        s3.put_object(Bucket=S3_BUCKET, Key=object_key, Body=json_bytes)
        print(f"S3 output log stored: {s3_uri}")
        return s3_uri
    except ClientError as e:
        print(f"S3 output log error: {e}")
        return ""
