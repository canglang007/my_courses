import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# 读取imdb数据集，继承dataset类
class IMDBDataset(Dataset):
    def __init__(self, data_dir, num_steps=500):
        self.num_steps = num_steps
        
        self.vocab = self.build_vocab(data_dir)
        self.train_data, self.test_data = self.read_imdb(data_dir)
    # 建立词汇表
    def build_vocab(self, data_dir):
        # Tokenization and vocabulary building
        all_tokens = []
        for split in ['train', 'test']:
            # 分消极和积极
            for sentiment in ['pos', 'neg']:
                folder_path = os.path.join(data_dir, split, sentiment)
                for filename in tqdm(os.listdir(folder_path), desc=f"Tokenizing {split} {sentiment}"):
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        tokens = text.split()  
                        all_tokens.extend(tokens)
        vocab = Counter(all_tokens)
        # 记录出现频率大于等于五次的
        vocab = [token for token, freq in vocab.items() if freq >= 5]  
        return {token: idx + 1 for idx, token in enumerate(vocab)}  
    # 读取词汇表的长度
    def get_vocab(self, data_dir):
        return len(self.build_vocab(data_dir))

    def get_vocab_list(self):
        return list(self.vocab.keys())
    # 读取imdb数据集
    def read_imdb(self, data_dir):
        train_data, test_data = [], []
        for split in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                folder_path = os.path.join(data_dir, split, sentiment)
                for filename in tqdm(os.listdir(folder_path), desc=f"Reading {split} {sentiment}"):
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        tokens = text.split() 
                        indexed_tokens = [self.vocab.get(token, 0) for token in tokens]  # Converting tokens to indices
                        indexed_tokens = self.truncate_pad(indexed_tokens)  # Truncating/padding to num_steps
                        # 打标签，当积极被记作1，消极被记作0
                        if split == 'train':
                            train_data.append((indexed_tokens, 1 if sentiment == 'pos' else 0))
                        else:
                            test_data.append((indexed_tokens, 1 if sentiment == 'pos' else 0))
        return train_data, test_data

    def truncate_pad(self, indexed_tokens):
        if len(indexed_tokens) < self.num_steps:
            indexed_tokens += [0] * (self.num_steps - len(indexed_tokens))  # Padding
        else:
            indexed_tokens = indexed_tokens[:self.num_steps]  # Truncating
        return indexed_tokens

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, index):
        if index < len(self.train_data):
            data, label = self.train_data[index]
        else:
            data, label = self.test_data[index - len(self.train_data)]
        return torch.tensor(data), torch.tensor(label).float()
# 写dataloader的部分    
def get_dataloader(data_dir, batch_size=32, num_steps=500, val_split=0.2):
    # 载入数据集
    imdb_dataset = IMDBDataset(data_dir, num_steps)

    # Splitting train_data into train and validation
    train_data = imdb_dataset.train_data
    train_data, val_data = train_test_split(train_data, test_size=val_split, random_state=42)

    train_dataset = torch.tensor([data for data, _ in train_data])
    train_labels = torch.tensor([label for _, label in train_data])

    val_dataset = torch.tensor([data for data, _ in val_data])
    val_labels = torch.tensor([label for _, label in val_data])

    test_dataset = torch.tensor([data for data, _ in imdb_dataset.test_data])
    test_labels = torch.tensor([label for _, label in imdb_dataset.test_data])
    # 训练部分的loader
    train_loader = DataLoader(TensorDataset(train_dataset, train_labels), batch_size=batch_size, shuffle=True)
    # 验证部分的loader
    val_loader = DataLoader(TensorDataset(val_dataset, val_labels), batch_size=batch_size, shuffle=False)
    # 测试部分的loader
    test_loader = DataLoader(TensorDataset(test_dataset, test_labels), batch_size=batch_size, shuffle=False)
    # 除了返回三个loader还有词汇表的长度和keys
    return train_loader, val_loader, test_loader, imdb_dataset.get_vocab(data_dir), imdb_dataset.get_vocab_list()

if __name__ == '__main__':
    # Usage example:
    data_dir = './aclImdb'
    batch_size = 32
    num_steps = 500
    # 划分验证集比例
    val_split = 0.1

    train_loader, val_loader, test_loader, i,my_dict = get_dataloader(data_dir, batch_size, num_steps, val_split)

    # 查看读取是否正确
    # for i, batch in enumerate(train_loader):
    #     input_data, labels = batch
    #     if len(input_data) == 0:
    #         print(len(input_data[0]))
    #         print(len(labels))
    #         print(input_data)
    #         print(labels)

    # print(len(val_loader))

    # for i, batch in enumerate(val_loader):
    #     input_data, labels = batch
    #     if i == 0:
    #         print(len(input_data[0]))
    #         print(len(labels))
    #         print(input_data)
    #         print(labels)


    # print(len(test_loader))

    # for i, batch in enumerate(test_loader):
    #     input_data, labels
    #     if i == 0:
    #         print(len(input_data[0]))
    #         print(len(labels))
    #         print(input_data)
    #         print(labels)
        
