from .upload.base import VideoCombineAndUpload, ImageUpload

NODE_CLASS_MAPPINGS = {
    "TMKK-UploadData": VideoCombineAndUpload,
    "TMKK-UploadImage": ImageUpload
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TMKK-UploadData": "TMKK-UploadData",
    "TMKK-UploadImage": "TMKK-UploadImage"
}
