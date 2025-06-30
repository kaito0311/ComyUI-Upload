import os
import shutil
import uuid
import subprocess

import torch
import numpy as np
from PIL import Image

import folder_paths


class VideoFormat:
    def __init__(self):
        self.name = ""
        self.infor = {}

    def get_infor(self):
        return self.infor


class H264Format(VideoFormat):
    name = "H.264"
    extension = "mp4"
    infor = {
        "main_pass":
        [
            "-n", "-c:v", "libx264",
            "-pix_fmt", ["pix_fmt", ["yuv420p", "yuv420p10le"]],
            "-crf", ["crf", "INT", {"default": 19,
                                    "min": 0, "max": 100, "step": 1}],
            "-vf", "scale=out_color_matrix=bt709",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"
        ],
        "fake_trc": "bt709",
        "audio_pass": ["-c:a", "aac"],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
        "trim_to_audio": ["trim_to_audio", "BOOLEAN", {"default": True}],
        "extension": "mp4"
    }


def tensor_to_int(tensor, bits):
    # TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))


def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)


class VideoCombineAndUpload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 25, "min": 15, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "video_output"}),
                "minio_endpoint": ("STRING", {"default": ""}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "bucket_name": ("STRING", {"default": "comfyui-videos"}),
                "folder_name": ("STRING", {"default": "videos"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("URL",)
    OUTPUT_NODE = True
    CATEGORY = "TMKK-UploadData"
    FUNCTION = "combine_video"

    def export_to_mp4(self, images, framerate, width, height, output_file):
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", "rawvideo",  # Input format: raw video
            "-vcodec", "rawvideo",  # Input codec: raw video
            "-s", f"{width}x{height}",  # Frame size
            "-pix_fmt", "rgb24",  # Input pixel format (RGB, 3 bytes per pixel)
            "-r", str(framerate),  # Framerate
            "-i", "-",  # Read input from stdin
            "-c:v", "libx264",  # H.264 codec for MP4
            "-pix_fmt", "yuv420p",  # Output pixel format for compatibility
            # Constant Rate Factor (quality: 0–51, lower is better)
            "-crf", "23",
            output_file
        ]

        # create folder if not exists
        output_folder = "/".join(output_file.split("/")[:-1])
        if os.path.exists(output_folder) is False:
            os.makedirs(output_folder, exist_ok=True)

        process = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Write each NumPy image to FFmpeg's stdin
        for img in images:
            # Ensure the image is in uint8 format and RGB
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Write raw RGB data to stdin
            process.stdin.write(img.tobytes())

        # Close the pipe and wait for FFmpeg to finish
        process.stdin.close()
        process.wait()

        # Check for errors
        if process.returncode != 0:
            error_output = process.stderr.read().decode()
            print(f"FFmpeg error: {error_output}")
        else:
            print(f"MP4 video saved as {output_file}")

    def upload_image_to_minio(self, image_path, folder, minio_endpoint, access_key, secret_key, bucket_name):
        import boto3
        from botocore.client import Config
        import os

        assert minio_endpoint is not None, "MINIO_ENDPOINT must be set"
        assert access_key is not None, "MINIO_ACCESS_KEY must be set"
        assert secret_key is not None, "MINIO_SECRET_KEY must be set"

        # Path to the file you want to upload
        # This is the key under which the file will be stored
        object_name = os.path.join(folder, os.path.basename(image_path))

        # Create S3 client for MinIO
        s3 = boto3.client('s3',
                          endpoint_url=minio_endpoint,
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key,
                          config=Config(signature_version='s3v4'),
                          region_name='us-east-1')  # Use any region, MinIO doesn't validate it

        # Upload the file
        s3.upload_file(image_path, bucket_name, object_name)
        print(
            f"✅ File '{image_path}' uploaded to '{bucket_name}/{object_name}'")

        return os.path.join(minio_endpoint, bucket_name, object_name)

    def combine_video(self, images,
                      frame_rate: int = 25,
                      filename_prefix: str = "video_output",
                      save_output: bool = True,
                      minio_endpoint: str = "",
                      access_key: str = "",
                      secret_key: str = "",
                      bucket_name: str = "comfyui-videos",
                      folder_name: str = "videos"
                      ):

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ("",)

        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        print(type(images))

        cpu_images = []
        for image in images:
            image = tensor_to_bytes(image)
            cpu_images.append(image)

        cpu_images = np.array(cpu_images).astype(np.uint8)
        images = cpu_images
        print(f"Shape of images: {images.shape}")
        print(f"Type of images: {type(images)}")
        first_image = images[0]
        width, height = first_image.shape[1], first_image.shape[0]
        output_file = os.path.join(
            full_output_folder, f"{filename}_{str(uuid.uuid4())}.mp4")

        self.export_to_mp4(images, frame_rate, width, height, output_file)

        if not save_output:
            # remove temp file
            shutil.rmtree(output_file, ignore_errors=True)

        print(f"Video saved to: {output_file}")

        if len(minio_endpoint) == 0 and len(access_key) == 0 and len(secret_key) == 0:
            return (output_file,)
        URL = self.upload_image_to_minio(
            output_file,
            folder_name,
            minio_endpoint,
            access_key,
            secret_key,
            bucket_name
        )

        return (URL,)


class ImageUpload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "image_output"}),
                "minio_endpoint": ("STRING", {"default": ""}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "bucket_name": ("STRING", {"default": "comfyui-images"}),
                "folder_name": ("STRING", {"default": "images"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("URL",)
    OUTPUT_NODE = True
    CATEGORY = "TMKK-UploadImage"
    FUNCTION = "upload_image"


    def upload_image_to_minio(self, image_path, folder, minio_endpoint, access_key, secret_key, bucket_name):
        import boto3
        from botocore.client import Config
        import os

        assert minio_endpoint is not None, "MINIO_ENDPOINT must be set"
        assert access_key is not None, "MINIO_ACCESS_KEY must be set"
        assert secret_key is not None, "MINIO_SECRET_KEY must be set"

        # Path to the file you want to upload
        # This is the key under which the file will be stored
        object_name = os.path.join(folder, os.path.basename(image_path))

        # Create S3 client for MinIO
        s3 = boto3.client('s3',
                          endpoint_url=minio_endpoint,
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key,
                          config=Config(signature_version='s3v4'),
                          region_name='us-east-1')  # Use any region, MinIO doesn't validate it

        # Upload the file
        s3.upload_file(image_path, bucket_name, object_name)
        print(
            f"✅ File '{image_path}' uploaded to '{bucket_name}/{object_name}'")

        return os.path.join(minio_endpoint, bucket_name, object_name)

    def upload_image(self, images,
                     filename_prefix: str = "image_output",
                     minio_endpoint: str = "",
                     access_key: str = "",
                     secret_key: str = "",
                     bucket_name: str = "comfyui-images",
                     folder_name: str = "images"
                     ):

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ("",)

        # get output information
        output_dir = folder_paths.get_output_directory()
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        # save image to file
        cpu_images = []
        for image in images:
            image = tensor_to_bytes(image)
            cpu_images.append(image)

        cpu_images = np.array(cpu_images).astype(np.uint8)
        images = cpu_images

        output_file = os.path.join(
            full_output_folder, f"{filename}_{str(uuid.uuid4())}.png")

        first_image = images[0]
        first_image = Image.fromarray(first_image).save(output_file)

        if len(minio_endpoint) == 0 and len(access_key) == 0 and len(secret_key) == 0:
            return (output_file,)

        URL = self.upload_image_to_minio(
            output_file,
            folder_name,
            minio_endpoint,
            access_key,
            secret_key,
            bucket_name
        )

        return (URL,)
