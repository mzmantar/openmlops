import os
from typing import Dict

import boto3


def upload_directory_to_minio(
    local_dir: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    prefix: str = "",
) -> Dict[str, str | int]:
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )

    try:
        client.head_bucket(Bucket=bucket_name)
    except Exception:
        client.create_bucket(Bucket=bucket_name)

    uploaded_count = 0
    normalized_prefix = prefix.strip("/")

    for root, _, files in os.walk(local_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, local_dir).replace("\\", "/")
            key = f"{normalized_prefix}/{rel_path}" if normalized_prefix else rel_path
            client.upload_file(full_path, bucket_name, key)
            uploaded_count += 1

    return {
        "bucket": bucket_name,
        "prefix": normalized_prefix,
        "uploaded_files": uploaded_count,
    }
