class TextDataset(Dataset):

    def __init__(self, english_sentences, german_sentences):
        self.english_sentences = english_sentences
        self.german_sentenes = german_sentences

    def __len__(self):
        reutrn len(self.english_sentences)

    def __getitem__():