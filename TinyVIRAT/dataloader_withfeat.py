import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import random
import numpy as np
from configuration import build_config
from tqdm import tqdm
import time
import warnings


def resize(frames, size, interpolation='bilinear'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(frames.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(frames, size=size, scale_factor=scale, mode='bicubic', align_corners=False)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TinyVirat(Dataset):
    def __init__(self, cfg, data_split, data_percentage, num_frames, skip_frames, input_size, shuffle=False):
        self.data_split = data_split
        self.num_classes = cfg.num_classes
        self.class_labels = [k for k, v in sorted(json.load(open(cfg.class_map, 'r')).items(), key=lambda item: item[1])]
        assert data_split in ['train', 'val', 'test']
        if data_split == 'train':
            annotations = json.load(open(cfg.train_annotations, 'r'))
            self.data_folder_lr = os.path.join(cfg.data_folder, 'cls_train')
            self.data_folder_sr = os.path.join(cfg.data_folder, 'SR_train')
        elif data_split == 'val':
            annotations = json.load(open(cfg.val_annotations, 'r'))
            self.data_folder_lr = os.path.join(cfg.data_folder, 'val')
            self.data_folder_sr = os.path.join(cfg.data_folder, 'SR_val')
        else:
            annotations = json.load(open(cfg.test_annotations, 'r'))
            self.data_folder = os.path.join(cfg.data_folder, 'val')
        self.annotations  = {}
        for annotation in annotations:
                     
            if annotation['id'] not in self.annotations:
                self.annotations[annotation['id']] = {}
            self.annotations[annotation['id']]['path'] = annotation['path']
            
            if data_split == 'test':
                self.annotations[annotation['id']]['label'] = []
            else:
                self.annotations[annotation['id']]['label'] = annotation['label']

        self.video_ids = list(self.annotations.keys())
        if shuffle:
            random.shuffle(self.video_ids)
        len_data = int(len(self.video_ids) * data_percentage)
        self.video_ids = self.video_ids[0:len_data]
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(),self.resize, self.normalize])

    def __len__(self):
        return len(self.video_ids)

    def _get_frames_list(self, frame_count):
        if frame_count < self.num_frames:
            list_1 = [x for x in range(frame_count)]
            list_2 = [frame_count - 1] * (self.num_frames - frame_count)
            list_1.extend(list_2)
            return list_1
        else:
            skip_frames = int(frame_count / self.num_frames)
            random_start = random.randint(0, frame_count - self.num_frames * skip_frames)
            frame_indicies = [random.randint(random_start + indx * skip_frames, random_start + (indx + 1) * skip_frames - 1) for indx in range(self.num_frames) ]
            return frame_indicies

    
    def load_frames_random(self, video_path):
        """Load frames in the given indices.
        """

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_frames_list(total_frames)
        frames = []
        for i in range(len(frame_indices)):
            cap.set(1, frame_indices[i])
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                warnings.warn(f'Fail to load frame {frame_indices[i]}')
        cap.release()
        return torch.from_numpy(np.stack(frames))

    def load_all_frames(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        ret = True
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        return frames

    def build_random_clip(self, video_path):
        frames = self.load_frames_random(video_path)
        frames = self.transform(frames)
        return frames

    def test_sample(self, video_path, sample_time = 5):
        frames = self.load_all_frames(video_path)
        frame_count = len(frames)
        if frame_count <= self.num_frames:
            frames = self.load_frames_random(video_path)
            frames = self.transform(frames)
            frames = torch.stack([frames])
            return frames
        frame_gap = int(np.floor(frame_count / self.num_frames))
        start_max = frame_gap + frame_count % self.num_frames
        start_indx = np.linspace(0, start_max - 1, num = sample_time, dtype = np.int64)
        all_frames = []
        for start in start_indx:
            clip_list = [start + i * frame_gap for i in range(self.num_frames)]
            frame_part = [frames[i] for i in clip_list]
            frame_part = torch.from_numpy(np.stack(frame_part))
            frame_part = self.transform(frame_part)
            all_frames.append(frame_part)
        all_frames = torch.stack(all_frames)
        return all_frames

    def sample_n_time(self, video_path, sample_time = 5):
        clips = torch.stack([self.transform(self.load_frames_random(video_path)) for _ in range(sample_time)])
        return clips

    def build_consecutive_clips(self, video_path):
        frames = self.load_all_frames(video_path)
        if len(frames) % self.num_frames != 0:
            frames = frames[:len(frames) - (len(frames) % self.num_frames)]
        clips = torch.stack([self.transform(x) for x in chunks(frames, self.num_frames)])
        return clips
    
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        lr_video_path = os.path.join(self.data_folder_lr ,self.annotations[video_id]['path'])
        sr_video_path = os.path.join(self.data_folder_sr ,self.annotations[video_id]['path'])
        
        if self.data_split == 'test':
            video_labels = []
        else:
            video_labels = self.annotations[video_id]['label']
        if self.data_split == 'train':
            feature_path = './feature/'+self.annotations[video_id]['path'][:-4]

            lr_clips = self.build_random_clip(lr_video_path)
            sr_clips = self.build_random_clip(sr_video_path)
        elif self.data_split == 'val':
            feature_path = './val_feature/' + self.annotations[video_id]['path'][:-4]
            lr_clips = self.build_random_clip(lr_video_path)
            sr_clips = self.build_random_clip(sr_video_path)
        else:
            lr_clips = self.test_sample(lr_video_path)
            sr_clips = self.test_sample(sr_video_path)
 
        if self.data_split == 'test':
            return lr_clips, sr_clips, [self.annotations[video_id]]
                
        label = np.zeros(self.num_classes)
        for _class in video_labels:
            label[self.class_labels.index(_class)] = 1
        bert_feat = np.load(feature_path+'/output.npy')
        return lr_clips, sr_clips, label, lr_video_path, bert_feat


if __name__ == '__main__':
    shuffle = True
    batch_size = 128

    dataset = 'TinyVirat'
    cfg = build_config(dataset)
    data_generator = TinyVirat(cfg, 'train', 1.0, num_frames=8, skip_frames=2, input_size=224)
    dataloader = DataLoader(data_generator, batch_size, shuffle=shuffle, num_workers=32)

    start = time.time()

    for epoch in range(0, 1):
        for i, (lr_clips, sr_clips, label, _, bert_feat) in enumerate(tqdm(dataloader)):
            
            lr_clips = lr_clips.data.numpy()
            sr_clips = sr_clips.data.numpy()
    print("time taken : ", time.time() - start)
