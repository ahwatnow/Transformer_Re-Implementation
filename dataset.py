from torch.utils.data import Dataset, DataLoader

class IwsltDataset(Dataset):

    def __init__(self, english_sentences, german_sentences):
        self.english_sentences = english_sentences
        self.german_sentences = german_sentences
    
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        return self.english_sentences[idx], self.german_sentence[idx]