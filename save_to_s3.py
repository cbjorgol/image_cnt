import os
import boto3
import config as cfg

def push_to_s3(file, bucket):
    s3 = boto3.resource('s3')
    for s3_bucket in s3.buckets.all():
        print(s3_bucket.name)

    data = open(file, 'rb')
    s3.Bucket(bucket).put_object(Key=file, Body=data)

models = os.listdir(cfg.MODEL_PATH)
for model in models:
    model_path = os.path.join(cfg.MODEL_PATH, model)
    if os.path.isfile(model_path):
        print("Pushing:", model_path)
        push_to_s3(model_path, 'cbjorgol-1')
    else:
        print("Not File:", model_path)

