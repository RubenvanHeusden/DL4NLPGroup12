import numpy as np
class GloveEmbeddings:
    """
    The GloveEmbedding class manages the loading, storage and retrieval
    of word embeddings from the GloVe project.

    attributes
    ----------
    self.embeds: dict
        stores the word embeddings as {word (string): embedding: np array }
        pairs

    self.embed_size: int
        represents the size of the word embeddings, this is mainly
        used to return random np arrays of the correct shape in case of
        OOV words

    methods
    ----------
    load_embeddings(filepath)
        loads the embeddings from the file indicated by the string filepath
        into the self.embeds dictionary

    get(word)
        attempts to fetch the embedding for the string word from the
        self.embeds dict, if it does not exist this method returns a random
        np array with uniform values between 0 and 1 of the appropriate size

    """
    def __init__(self, embed_size):
        """
        :param embed_size: an integer specifying the size of the word embeddings
        this should be equal to the size of the embeddings from the
        embeddings file
        """

        self.embeds = {}
        self.embed_size = embed_size
        self.embed_matrix = None

    def load_embeddings(self, filepath):
        """

        :param filepath: string containing the filepath of the embedding
        file that should be loaded in
        """
        with open (filepath, 'r',encoding='utf-8') as f_obj:
            for line in f_obj:
                word, *embedding = line.split()
                self.embeds[word] = np.array(embedding, dtype=float)
        return None

    def get(self, word):
        """

        :param word: a string for which the word embedding will be looked up
        in self.embeds
        :return: the word embedding for word if it exists, returns a random
        uniform np array of size self.embed_size otherwise
        """
        embedding = self.embeds.get(word, np.random.uniform(size=self.embed_size))

        return embedding

    def set_embedding_matrix(self, word_dict):
        self.embed_matrix = np.zeros((len(word_dict), self.embed_size))
        for word, idx in word_dict.items():
            self.embed_matrix[idx] = self.get(word)



