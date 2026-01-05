import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import cv2
from PIL import Image
import os
import numpy as np
from torchvision import transforms

import random
# Path to the UTKFace dataset directory
data_dir = "/path/to/UTKFace/"

# Ethnicity labels (0=White, 1=Black, 2=Asian, 3=Indian)
ethnicities = ['White', 'Black', 'Asian', 'Indian']

# Age labels (0=ages 0-19, 1=ages 20-40, 2=ages 40+)
age_ranges = [0, 1, 2]

# Create a custom dataset class for the UTKFace dataset
def load_dataset(data_dir):

        age_list = []
        gender_list = []
        race_list = []
        images = []
        # Loop over each image file in the directory
        L = len(os.listdir(data_dir))
        cnt = L //200 
        for j, filename in enumerate(os.listdir(data_dir)):
            # Extract the age and ethnicity information from the filename
            information = filename.split("_")
            if len(information)<4:
                continue
            age, gender, race, _ = information
            age = int(age)
            race = int(race)

            # Exclude images from the "Others" ethnic group
            if race == 4:
                continue

            # Assign the appropriate age range label
            if age < 20:
                age_label= 0
            elif age < 40:
                age_label = 1
            else:
                age_label = 2
            age_list.append(age_label)
            gender_list.append(int(gender))
            race_list.append(int(race))
            # Load the image and append it to the images list
            image = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            image = np.asarray(image)
            # print('shape',image.shape)
            images.append(image)
            if (j+1) % cnt == 0:
                print(f"Loaded {j+1}/{L} images")

            # Create a label by combining the ethnicity and age range labels
            # label = ethnicity * len(age_ranges) + age_range
        return images, age_list, race_list, gender_list
        # return images, age_list, gender_list, gender_list


class TensorDataset(Dataset):
    def __init__(self, images, labels, color, gender, transform): # images: n x c x h x w tensor
        self.images = images
        self.labels = labels
        self.color = color
        self.gender = gender
        self.transform = transform
    def __getitem__(self, index):
        if self.transform:
        # return self.images[index], self.labels[index], self.color[index], self.gender[index]
            return self.transform(self.images[index]), self.labels[index], self.color[index]

    def __len__(self):
        return self.images.shape[0]

def UTKFaceDataset(data_dir="./data/UTKFace", transform=None):
    data_dir_1 = '/mnt/DatasetCondensation-master/data/UTKFace'
    data_dir_2 = '/remote-home/iot_fangshenhao/DatasetCondensation-master/data/UTKFace'
    data_dir_3 = '/remote-home/share/Fisher1/DatasetCondensation-master/data/UTKFace'
    data_dir_3 = '/tmp/FairDD/data/UTKFace'

    if os.path.exists(data_dir_1):
        data_dir = data_dir_1
    elif os.path.exists(data_dir_2):
        data_dir = data_dir_2
    elif os.path.exists(data_dir_3):
        data_dir = data_dir_3

    try: 
        loaded_data = torch.load(data_dir + '/dataset_checkpoint.pt', weights_only=False)
        images = loaded_data['images']
        labels = loaded_data['labels']
        color = loaded_data['color']
        gender = loaded_data['gender']
    
    except:
        images, labels, color, gender = load_dataset(data_dir)
        data_to_save = {
        'images': images,
        'labels': labels,
        'color': color,
        'gender': gender
        }
        torch.save(data_to_save, data_dir + '/dataset_checkpoint.pt')

    data_len = len(images)

    test_size = int(0.2 * data_len)

    label_idx_dict = {}
    color_idx_dict = {}

    for labels_id in np.unique(labels):
        print("labels",labels_id)
        label_idx_dict[labels_id] = [idx for idx, c in enumerate(labels) if labels_id == c]
    for color_id in np.unique(color):
        print("color_id",color_id)
        color_idx_dict[color_id] = [idx for idx, c in enumerate(color) if color_id == c]
    # sample_num = int(test_size/(len(color_idx_dict) * len(label_idx_dict)))
    sample_num = 100
    for i in range(len(label_idx_dict)):
        for j in range(len(color_idx_dict)):
            intersection = list(set(label_idx_dict[i]) & set(color_idx_dict[j]))
            sample_num = min(len(intersection), sample_num)


    test_idx_dict = {}
    test_idx = []
    for i in range(len(label_idx_dict)):
        for j in range(len(color_idx_dict)):
            intersection = list(set(label_idx_dict[i]) & set(color_idx_dict[j]))
            # print(i,j, sample_num, len(intersection))
            # select_idx = random.sample(intersection,  min(len(intersection), sample_num))
            select_idx = random.sample(intersection, sample_num)
            print("test",i,j,len(select_idx))
            test_idx.extend(select_idx)
            test_idx_dict[(i,j)] = select_idx
            label_idx_dict[i] = set(label_idx_dict[i]) - set(select_idx)
            color_idx_dict[j] = set(color_idx_dict[j]) - set(select_idx)

    train_idx = {}
    for i in range(len(label_idx_dict)):
        for j in range(len(color_idx_dict)):
            intersection = list(set(label_idx_dict[i]) & set(color_idx_dict[j]))
            print("train",i,j,len(intersection))
            train_idx[(i, j)] = intersection

    new_train_idx = np.concatenate([np.array(v) for k, v in train_idx.items()])
    new_train_idx = new_train_idx.astype(int)


    total_idx = list(range(data_len))
    train_idx = list(set(total_idx) - set(test_idx))

    # print(data_len, len(train_idx), len(test_idx), len(train_idx)/len(test_idx))

    images = np.asarray(images)
    labels = np.asarray(labels)
    color = np.asarray(color)
    gender = np.asarray(gender)

    mean = (0.6135867892520825, 0.4722519020075052, 0.4070309090621279)
    std = (0.25492949231398077, 0.23054187352850616, 0.22834804939251105)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    # print(new_train_idx)
    train_dataset = TensorDataset(images[new_train_idx], labels[new_train_idx], color[new_train_idx], gender[new_train_idx], transform)
    test_dataset = TensorDataset(images[test_idx], labels[test_idx], color[test_idx], gender[test_idx], transform)
    # print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset, mean, std


if __name__ == "__main__":
    a = UTKFaceDataset()