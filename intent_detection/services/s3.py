"""
    intent_detection.s3
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Client that downloads and uploads intent detection model to S3 (or to local folder in "offline" mode)
    @author: Daria Ozerova
"""
import pathlib
from typing import Optional

import pickle
import boto3

from botocore import exceptions

from intent_detection import (
    DIR_DATA,
    S3_BUCKET,
    MODEL_OBJECT_NAME,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
)


s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)


def put_object(bucket_name: str, obj_name: str, obj: bytes):
    """
    Upload an object to S3
    :param bucket_name: name of S3 bucket
    :param obj_name: name of an object
    :param obj: object in bytes
    """
    try:
        s3.put_object(Bucket=bucket_name, Key=obj_name, Body=obj)
    except exceptions.ClientError as exception:
        print(f"Unable to upload object {MODEL_OBJECT_NAME} to S3: {exception}")


# TODO: redis (or something else) cache
def get_object(bucket_name: str, obj_name: str) -> Optional[bytes]:
    """
    Download an object from S3 if exists
    :param bucket_name: name of S3 bucket
    :param obj_name: name of an object
    :return: object as a sequence of bytes
    """
    try:
        response = s3.get_object(Bucket=bucket_name, Key=obj_name)
        return response["Body"].read()
    except exceptions.ClientError as exception:
        print(f"Unable to download object {MODEL_OBJECT_NAME} from S3: {exception}")
    return None


def put_object_offline(obj_name: str, obj: bytes):
    """
    Simulates the upload to S3
    :param obj_name: name of an object
    :param obj: object in bytes
    """
    path = DIR_DATA / obj_name
    data = pickle.loads(obj)
    with path.open("wb") as pickle_file:
        pickle.dump(data, pickle_file)


def get_object_offline(obj_name: str) -> bytes:
    """
    Simulates the download of an object from S3 if exists
    :param obj_name: name of an object
    :return: object as a sequence of bytes
    """
    path = DIR_DATA / obj_name
    if path.exists():
        with path.open("rb") as pickle_file:
            # pickle.load(pickle_file)
            return pickle.dumps(pickle.load(pickle_file))
    return bytes()


def upload_model(model: dict, mode: str = "offline"):
    """
    Upload intent detection model to S3
    :param model: dictionary containing model parameters
    :param mode: "offline" or "online" determines whether to call aws
    """
    if mode == "online":
        put_object(
            bucket_name=S3_BUCKET, obj_name=MODEL_OBJECT_NAME, obj=pickle.dumps(model)
        )
    else:
        put_object_offline(obj_name=MODEL_OBJECT_NAME, obj=pickle.dumps(model))


def download_model(mode: str = "offline") -> dict:
    """
    Download intent detection model from S3
    :param mode: "offline" or "online" determines whether to call aws
    :return: dictionary containing model parameters
    """
    if mode == "online":
        if obj := get_object(bucket_name=S3_BUCKET, obj_name=MODEL_OBJECT_NAME):
            return pickle.loads(obj)
    else:
        if obj := get_object_offline(obj_name=MODEL_OBJECT_NAME):
            return pickle.loads(obj)
    return {}


def upload_file(path: str, bucket_name: str = None):
    """
    Upload a file to an S3 bucket
    :param bucket_name: name of S3 bucket
    :param path: path to a file
    """
    bucket_name = bucket_name or S3_BUCKET
    file_name = pathlib.Path(path).name
    try:
        s3.upload_file(path, bucket_name, file_name)
        print("Uploaded file: %s", file_name)
    except exceptions.ClientError as exception:
        print(exception)


def download_file(path, bucket_name=None):
    """
    Download a file from an S3 bucket
    :param bucket_name: name of S3 bucket
    :param path: path where a file will be saved
    """
    bucket_name = bucket_name or S3_BUCKET
    file_name = pathlib.Path(path).name
    try:
        s3.download_file(bucket_name, file_name, path)
        print("Downloaded file: %s", file_name)
    except exceptions.ClientError as exception:
        print(exception)


def generate_url(file):
    """
    Create url for accessing a file on S3 server
    :param file: name of a file on S3
    """
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": file}
    ).replace("&", "%26")
