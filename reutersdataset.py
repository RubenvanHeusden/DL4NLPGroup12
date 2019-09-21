import os
import torch
from nltk.tokenize import  word_tokenize
from torch.utils.data import Dataset
import numpy as np
from gloveembeddings import GloveEmbeddings


class ReutersDataset(Dataset):
    """
        The ReutersDataset manages the loading of the Reuters
        corpus, as well as loading in the labels for each document
        and pre processing the input documents

        attributes
        ------------
        self.mode: string either "training" or "test"
            this attribute sets the mode of the dataset, which is used
            to determine which dataset to load, either train or test
            and load the corresponding document labels for it

        methods
        ------------
        _read_data()
            this method reads the data in the files in either the training
            or test folders and does some pre processing on the data such as
            converting tokenizing all documents and converting all text
            into lowercase

        _construct_labels(label_filepath)
            this method reads the data stored in the category file and
            links every file with its target labels, storing them in a
            dictionary as {file_id (string) : target_labels (list)}

    """
    def _read_data(self):
        """

        :return: list of documents where each entry in the list is itself
        a list containing the tokens/words in each document
        """
        documents = []
        for i in range(len(self.data_files)):
            with open(self.root_path+"/"+self.mode+"/"+self.data_files[i], 'r') as f:
                text = word_tokenize(f.read().lower())
                documents.append(text)
        return documents

    def _contruct_labels(self, label_filepath):
        """
        :param label_filepath: string that represents the file path of the
        cats.txt file relative to the main reuters/folder
        :return: dictionary containing file_ids linked with categories
        """
        file_dict = {}
        with open(label_filepath, 'r') as f:
            for line in f:
                filename, *cats = line.split()
                file_dict[filename] = cats
        return file_dict

    def __init__(self, root_path, mode='train'):
        """
        parameters
        -----------
        root_path (string)
            this string represents the location of the main reuters
            data folder, this should be either a relative or absolute path.

        mode (string)
            this string sets the mode of the dataloading, if it is set to train
            the class will load the training texts and labels, if it is set
            to test it will load the test documents and labels. This
            matters in particular for the __getitem__ function as this
            will return different data depending on the mode
        """
        if mode == 'train':
            self.mode='training'
        elif mode == 'test':
            self.mode = 'test'


        self.root_path = root_path
        self.data_files = os.listdir(self.root_path+"/"+self.mode)

        self.documents = self._read_data()
        self.vocab = set([word for sentence in self.documents for word in sentence])
        self.vocab_dict = {word:x for x,word in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab_dict)

        self.doc_labels = self._contruct_labels(root_path+"/"+"cats.txt")
        self.cats_unique = set(item for key,value in self.doc_labels.items() for
                            item in value)
        self.cats_dict = {item:x for x, item in enumerate(self.cats_unique)}
        self.num_cats = len(self.cats_dict)


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        :param idx: index of item that needs to be retrieved
        :return: document at specified location and the label(s) associated
        with it as a one hot vector
        """
        document = self.documents[idx]
        text_vec = np.array([self.vocab_dict[word] for word in document])

        labels = self.doc_labels[self.mode+"/"+self.data_files[idx]]
        label_vec = np.zeros((self.num_cats, 1))
        np.put(label_vec, [self.cats_dict[label] for label in labels],
               np.array([1]))
        return text_vec, label_vec

