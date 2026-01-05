import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob

class TensorDataset(Dataset):
    """
    A simple dataset that holds pre-loaded data in memory.
    Mimics the behavior of the original bFFHQDataset but uses in-memory arrays.
    """
    def __init__(self, images, target_attrs, bias_attrs, transform=None):
        self.images = images        # numpy array of shape (N, 64, 64, 3)
        self.target_attrs = target_attrs
        self.bias_attrs = bias_attrs
        self.transform = transform

    def __getitem__(self, index):
        # Retrieve image from memory
        img_arr = self.images[index]
        
        # Convert to PIL because transforms usually expect PIL or Tensor
        # The bffhq approach usually does ToPIL -> Resize -> ToTensor -> Normalize
        # We already resized during load, so we just convert.
        if self.transform is not None:
            image = self.transform(img_arr)
        else:
            image = img_arr

        first_attr = self.target_attrs[index]
        second_attr = self.bias_attrs[index]

        return image, first_attr, second_attr

    def __len__(self):
        return len(self.images)

def get_file_list(root, split):
    """Helper to get file paths based on split logic from original bFFHQ"""
    if split == '0.5pct':
        align = glob(os.path.join(root, split, 'align', "*", "*"))
        conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
        return align + conflict
    elif split == 'valid':
        return glob(os.path.join(root, split, "*"))
    elif split == 'test':
        return glob(os.path.join(root, split, "*"))
    return []

def load_images_and_labels(files, resize_dim=64):
    """
    Loads all images into a numpy array and extracts labels.
    Resizes immediately to save RAM.
    """
    images = []
    target_attrs = []
    bias_attrs = []
    
    print(f"Processing {len(files)} files...")
    
    for i, fpath in enumerate(files):
        try:
            # Extract attributes based on filename format: ..._target_bias.jpg
            # Original: first_attr = int(fpath.split('_')[-2])
            # Original: second_attr = int(fpath.split('_')[-1].split('.')[0])
            parts = fpath.split('_')
            first_attr = int(parts[-2])
            second_attr = int(parts[-1].split('.')[0])

            # Load image, convert to RGB, and resize immediately
            img = Image.open(fpath).convert('RGB').resize((resize_dim, resize_dim))
            
            images.append(np.asarray(img))
            target_attrs.append(first_attr)
            bias_attrs.append(second_attr)
            
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{len(files)} images")

    return np.array(images), np.array(target_attrs), np.array(bias_attrs)

def BFFHQ(data_dir="./data/bffhq"):

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Standard transformation for the training loop
    # Note: We use ToPILImage because our TensorDataset returns numpy arrays
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(64), # Already resized during loading
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
    ])

    # Handle directory overrides
    data_dir_3 = '/tmp/FairDD/data/bffhq'
    if os.path.exists(data_dir_3):
        data_dir = data_dir_3
    
    checkpoint_path = os.path.join(data_dir, 'bffhq_fast_checkpoint.pt')

    try:
        # Try loading pre-processed data
        print(f"Loading dataset from checkpoint: {checkpoint_path}")
        loaded_data = torch.load(checkpoint_path, weights_only=False)
        
        train_data = loaded_data['train']
        test_data = loaded_data['test']
        
        # Reconstruct datasets
        train_dataset = TensorDataset(
            train_data['images'], 
            train_data['targets'], 
            train_data['biases'], 
            transform=transform
        )
        test_dataset = TensorDataset(
            test_data['images'], 
            test_data['targets'], 
            test_data['biases'], 
            transform=transform
        )
        print("Checkpoint loaded successfully.")

    except Exception as e:
        print(f"Checkpoint not found or failed ({e}). Loading from raw files...")
        
        # 1. Load Train Data (Split: '0.5pct')
        train_files = get_file_list(data_dir, '0.5pct')
        train_imgs, train_targets, train_biases = load_images_and_labels(train_files)
        
        # 2. Load Test Data (Split: 'test')
        test_files = get_file_list(data_dir, 'test')
        test_imgs, test_targets, test_biases = load_images_and_labels(test_files)
        
        # 3. Create Datasets
        train_dataset = TensorDataset(train_imgs, train_targets, train_biases, transform=transform)
        test_dataset = TensorDataset(test_imgs, test_targets, test_biases, transform=transform)

        # 4. Save Checkpoint
        # We store the raw arrays, not the dataset objects, to keep pickle simple
        data_to_save = {
            'train': {
                'images': train_imgs,
                'targets': train_targets, 
                'biases': train_biases
            },
            'test': {
                'images': test_imgs, 
                'targets': test_targets, 
                'biases': test_biases
            }
        }
        
        # Ensure dir exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        torch.save(data_to_save, checkpoint_path)
        print(f"Saved dataset checkpoint to {checkpoint_path}")

    return train_dataset, test_dataset, mean, std

if __name__ == "__main__":
    # Test run
    tr_set, te_set, m, s = BFFHQ()
    print(f"Train set size: {len(tr_set)}")
    print(f"Test set size: {len(te_set)}")


# import torch
# import os
# from PIL import Image
# from torch.utils.data.dataset import Dataset
# from glob import glob

# class bFFHQDataset(Dataset):
#     target_attr_index = 0
#     bias_attr_index = 1

#     def __init__(self, root, split, transform=None):
#         super(bFFHQDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.target_attrs = []
#         self.bias_attrs = []

#         if split == '0.5pct':
#             self.align = glob(os.path.join(root, split, 'align', "*", "*"))
#             self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
#             self.data = self.align + self.conflict

#         elif split == 'valid':
#             self.data = glob(os.path.join(root, split, "*"))

#         elif split == 'test':
#             self.data = glob(os.path.join(root, split, "*"))

#         # 提取属性标签并保存到成员变量中
#         for fpath in self.data:
#             first_attr = int(fpath.split('_')[-2])
#             second_attr = int(fpath.split('_')[-1].split('.')[0])
#             self.target_attrs.append(first_attr)
#             self.bias_attrs.append(second_attr)

#         self.target_attrs = torch.LongTensor(self.target_attrs)
#         self.bias_attrs = torch.LongTensor(self.bias_attrs)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):

#         fpath = self.data[index]
#         # first_attr = int(fpath.split('_')[-2])
#         # second_attr = int(fpath.split('_')[-1].split('.')[0])
#         # attr = torch.LongTensor([first_attr, second_attr])
#         image = Image.open(fpath).convert('RGB')
#         print('image opened: ',fpath)

#         if self.transform is not None:
#             image = self.transform(image)

#         first_attr = self.target_attrs[index]
#         second_attr = self.bias_attrs[index]

#         # return image, attr
#         return image, first_attr, second_attr # first_attr label,second_attr sensitive
#         # return image, second_attr, first_attr # first_attr label,second_attr sensitive

# def BFFHQ(data_dir = "./data/bffhq"):

#     mean = [0.4914, 0.4822, 0.4465]
#     std = [0.2023, 0.1994, 0.2010]
#     from torchvision import transforms

#     transform = transforms.Compose([
#         transforms.Resize(64),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std), ])
#     # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


#     data_dir_3 = '/tmp/FairDD/data/bffhq'
#     if os.path.exists(data_dir_3):
#         data_dir = data_dir_3


#     try: 
#         loaded_data = torch.load(data_dir + '/bffhq_dataset_checkpoint.pt', weights_only=False)
#         train_dataset = loaded_data['train_dataset']
#         test_dataset = loaded_data['test_dataset']

#     except:
#         train_dataset = bFFHQDataset(root = data_dir, split = '0.5pct', transform=transform)
#         test_dataset = bFFHQDataset(root = data_dir, split = 'test', transform=transform)

#         data_to_save = {
#         'train_dataset': train_dataset,
#         'test_dataset': test_dataset,
#         }
#         torch.save(data_to_save, data_dir + '/bffhq_dataset_checkpoint.pt')



#     return train_dataset, test_dataset, mean, std