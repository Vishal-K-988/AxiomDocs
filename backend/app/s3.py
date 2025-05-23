import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()

# here is the S3 configurations 
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
     region_name=os.getenv('AWS_REGION', 'us-east-1')
)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# for security purposes I'm generating a presigned urk for each session 
def generate_presigned_url(s3_key: str, expiration: int = 3600) -> Optional[str]:
    
    try:
        response = s3_client.generate_presigned_url(
            'get_object',

            Params={
                'Bucket': BUCKET_NAME,
                'Key': s3_key
            }
             ,
            
            
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        print(f"Error in generating presigned URL: {e}")
        return None


# 1. Uploading file to s3 and returning the s3 key; basically we are returnig the url / path where the file is stored on the s3 bucket 3+

def upload_file_to_s3(file_data: bytes, 
                      file_extension: str, 
                      user_id: str = "temp_user"
                      ) -> Optional[str]:

    try:
        # Creating a user specific folder as we are integating the clerk authentication
        #adn with the help of htat we need to organize the file w.r.t each user ! 
        s3_key = f"users/{user_id}/{uuid.uuid4()}{file_extension}"

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_data
        )
        return s3_key
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        return None


# 2. Deleting file 
def delete_file_from_s3(s3_key: str) -> bool:

    try:
        s3_client.delete_object(
            Bucket=BUCKET_NAME,

            Key=  s3_key
        )

        return True
    except ClientError as e:
        print(f"Error deleting file from S3: {e}")
        return False

# 3. Renaming file 
def rename_file_in_s3(old_s3_key: str, new_filename: str, user_id: str) -> Optional[str]:
   
    try:
        # Geting  the file extension from the old key
        file_extension = os.path.splitext(old_s3_key)[1]

        # Create new S3 key with the new filename and user folder
        new_s3_key = f"users/{user_id}/{new_filename}{file_extension}"
        
        s3_client.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={'Bucket': BUCKET_NAME, 'Key': old_s3_key},
            Key=new_s3_key
        )
        
        s3_client.delete_object(
            Bucket=BUCKET_NAME,
            Key=old_s3_key
        )
        
        return new_s3_key
    
    except ClientError as e:
        print(f"Error renaming file in S3: {e}")
        return None 