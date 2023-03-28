from monai.data import Dataset
import nibabel as nib
import numpy as np
import os
import torch

class BraTS18Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', class_type=0):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_type 

        # Load the list of subjects
        if self.mode == 'train':
            self.subjects = self.get_subject_dirs()
        elif self.mode == 'test':
            self.subjects = os.listdir(os.path.join(self.root_dir, self.mode))

    def get_subject_dirs(self):
        dirs = []
        mode_dir = os.path.join(self.root_dir, self.mode)
        for class_dir in os.listdir(mode_dir):
            class_dir = os.path.join(mode_dir, class_dir)
            for subject_dir in os.listdir(class_dir):
                subject_dir = os.path.join(class_dir, subject_dir)
                dirs.append(f"{class_dir}/{subject_dir}")
        return dirs

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # Load the image and label data for the given subject
        if self.mode == "train":
            data = []
            class_dir, subject_id = self.subjects[idx].split("/")
            image_paths = [f"{self.root_dir}/{self.mode}/{class_dir}/{subject_id}/{subject_id}_t1.nii.gz",
                           f"{self.root_dir}/{self.mode}/{class_dir}/{subject_id}/{subject_id}_t1ce.nii.gz",
                           f"{self.root_dir}/{self.mode}/{class_dir}/{subject_id}/{subject_id}_t2.nii.gz",
                           f"{self.root_dir}/{self.mode}/{class_dir}/{subject_id}/{subject_id}_flair.nii.gz"]
            label_path = f"{self.root_dir}/{self.mode}/{class_dir}/{subject_id}/{subject_id}_seg.nii.gz"
            for image_path in image_paths:
                image = nib.load(image_path).get_fdata(dtype=np.float32)
                data.append(image)
            label = nib.load(label_path).get_fdata().astype(np.float32)
        elif self.mode == "test":
            data = []
            subject_id = self.subjects[idx]
            image_paths = [f"{self.root_dir}/{self.mode}/{subject_id}/{subject_id}_t1.nii.gz",
                        f"{self.root_dir}/{self.mode}/{subject_id}/{subject_id}_t1ce.nii.gz",
                        f"{self.root_dir}/{self.mode}/{subject_id}/{subject_id}_t2.nii.gz",
                        f"{self.root_dir}/{self.mode}/{subject_id}/{subject_id}_flair.nii.gz"]
            label_path = f"{self.root_dir}/{self.mode}/{subject_id}/{subject_id}_seg.nii.gz"
            for image_path in image_paths:
                image = nib.load(image_path).get_fdata(dtype=np.float32)
                data.append(image)
            label = nib.load(label_path).get_fdata().astype(np.float32)
        
        data = np.stack(data, axis=0)
        data = torch.from_numpy(data)
        label = torch.from_numpy(data)
        # Apply transforms
        if self.transform:
            image, label = self.transform(image, label)

        return {'image': image, 'label': label}