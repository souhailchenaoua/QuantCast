# utils/utils_s3.py
try:
    import boto3
except ImportError:
    boto3 = None

def s3_upload_if_configured(local_path: str, key: str):
    """Upload file to S3 if boto3 + env config present; else do nothing."""
    if boto3 is None:
        print(f"[utils_s3] boto3 not installed, skipping S3 upload of {local_path}")
        return
    import os
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        print(f"[utils_s3] no S3_BUCKET configured, skipping upload {local_path}")
        return
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","us-east-1"))
    s3.upload_file(local_path, bucket, key)
    print(f"[utils_s3] uploaded {local_path} to s3://{bucket}/{key}")
