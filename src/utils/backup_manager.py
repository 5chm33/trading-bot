import os
import shutil
from datetime import datetime
import boto3  # For cloud backups

class BackupManager:
    def __init__(self, local_dir="backups", cloud_bucket=None):
        self.local_dir = local_dir
        self.s3 = boto3.client('s3') if cloud_bucket else None
        self.cloud_bucket = cloud_bucket

    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = f"{self.local_dir}/backup_{timestamp}"

        # Local backup
        os.makedirs(f"{backup_path}/logs", exist_ok=True)
        shutil.copytree("logs", f"{backup_path}/logs", dirs_exist_ok=True)
        shutil.copytree("models", f"{backup_path}/models", dirs_exist_ok=True)

        # Cloud sync if configured
        if self.s3:
            self._upload_to_cloud(backup_path)

        return backup_path

    def _upload_to_cloud(self, path):
        for root, _, files in os.walk(path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.relpath(local_path, path)
                self.s3.upload_file(local_path, self.cloud_bucket, s3_path)

# Add to nightly_routine()
def nightly_routine():
    backup = BackupManager(cloud_bucket="your-bucket-name")
    backup.create_backup()
