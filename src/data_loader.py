import av

import torch
from torchvision import transforms
from torchvision.datasets import UCF101
from torch.utils.data.distributed import DistributedSampler

import cv2
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')

class UCF101_Dataset_Loader:
    def __init__(self: 'UCF101_Dataset_Loader', ucf_data_dir: 'str', ucf_label_dir: 'str', frames_per_clip: int, 
                 step_between_clips: int, height:int, width: int, batch_size: int, data_parallel_distributed_training: bool = False):
        self.ucf_data_dir = ucf_data_dir
        self.ucf_label_dir = ucf_label_dir
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        
        self.height = height
        self.width = width
        self.batch_size = batch_size
        
        self.data_parallel_distributed_training = data_parallel_distributed_training
        
        self.train_loader, self.test_loader, self.classes = self._data_loaer()
    
    def _collate(self: 'UCF101_Dataset_Loader', batch: torch.utils.data) -> torch.utils.data.dataloader.default_collate:
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)
    
    def _data_loaer(self: 'UCF101_Dataset_Loader') -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        
        transformer = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, H, W, C) to (C,T,H,W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, (self.height, self.width))),
            ])
        
        print('Loading Training Data .........')
        train_dataset = UCF101(self.ucf_data_dir, self.ucf_label_dir, frames_per_clip=self.frames_per_clip,
                       step_between_clips=self.step_between_clips, train=True, transform=transformer)
        if self.data_parallel_distributed_training:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                           collate_fn=self._collate, sampler=DistributedSampler(train_dataset))
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True,
                                           collate_fn=self._collate)
        
        print('Loading Testing Data .........')
        test_dataset = UCF101(self.ucf_data_dir, self.ucf_label_dir, frames_per_clip=self.frames_per_clip,
                      step_between_clips=self.step_between_clips, train=False, transform=transformer)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True,
                                          collate_fn=self._collate)
        
        print('\n')
        print(f"Total number of train samples: {len(train_dataset)}")
        print(f"Total number of test samples: {len(test_dataset)}")
        
        print(f"Total number of (train) batches: {len(train_loader)}")
        print(f"Total number of (test) batches: {len(test_loader)}")
        print('\n')
        print(f"Total Number of Classes: {len(train_dataset.classes)}")
        
        return train_loader, test_loader, train_dataset.classes
    
    

def load_video(video_path, frames_per_clip, step_between_clips, height, width, batch_size):
    # Initialize a list to store the clips
    clips = []

    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame
    for i in range(0, frame_count - frames_per_clip + 1, step_between_clips):
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        for j in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            resized_frame = cv2.resize(frame, (width, height))
            # Convert color from BGR to RGB
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if len(frames) == frames_per_clip:
            clips.append(np.stack(frames))

    cap.release()

    # Transformation function
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformation to each frame in each clip
    transformed_clips = []
    for clip in clips:
        transformed_frames = [transform(frame) for frame in clip]
        transformed_clip = torch.stack(transformed_frames)
        transformed_clips.append(transformed_clip)

    # Stack clips into a tensor
    clips_tensor = torch.stack(transformed_clips)

    # Reshape to get the required output format
    # The shape is [batch, channels, frames_per_clip, height, width]
    batch = clips_tensor.view(-1, 3, frames_per_clip, height, width)

    # Take only as many batches as needed
    return batch[:batch_size]