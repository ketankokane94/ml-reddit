def remove_words(words):
    return words

def get_unique_words(words):
    bag = []

    for word in words:
        if word not in bag:
            bag.append(word)

    return bag

def form_vectors(input, bag):
    sample_vectors = []

    for sample in input:
        vector = {}

        # Set initially all to 0
        for word in bag:
            vector[word] = 0

        # Update frequencies
        words = sample.split(' ')

        # We will find words that are in bag only
        # since we already processed these words
        for word in words:
            if word in bag:
                vector[word] += 1

        sample_vectors.append(vector)

    return sample_vectors

def bag_of_words(input):
    '''
        We form a bag with all words that exist in input and
        form a vector for each sample based on number of occurences
        of each word in bag.
    :param input: list of strings, each for one sample of reddit
    :return: list of dictionaries where each dictionary is a vector
            for a sample.
    '''
    # Get all words into a list
    words = []

    for string in input:
        sample_words = string.split(' ')
        for word in sample_words:
            words.append(word)

    #print(words)

    # Remove irrelevant words
    words = remove_words(words)

    # Get all unique words as
    bag = get_unique_words(words)

    vectors = form_vectors(input, bag)

    for vector in vectors:
        print(vector)
        print('\n')

    return vectors

def bag_predicted_data(training_vectors, input):
    '''
        We bag differently for data to be predicted since we
        consider features i.e. words already used only.
    :param vectors: dictionary with words used for training.
    :param input: list of strings for examples to be predicted.
    :return:
    '''
    # Get all words into a list
    words = []

    for string in input:
        sample_words = string.split(' ')
        for word in sample_words:
            words.append(word)

    #print(words)

    # We do not bother about irrelevant words since we use words
    # used while training so already removed irrelevant words.

    # Training vectors has keys as all the words used.
    # it is a list of vectors so need any one
    vectors = form_vectors(input, list(training_vectors[0].keys()))

    for vector in vectors:
        print(vector)
        print('\n')

    return vectors


def read_file(filename):
    with open(filename, 'r') as file:
        file_chars = ''

        for line in file:
            for char in line:
                file_chars += char.lower()

            # Last word will include \n so we remove it
            if '\n' in file_chars:
                # Would have been just added at the end
                file_chars = file_chars[:-1]

            file_chars += ' '

        # Extra space would have been added so we remove it
        file_chars = file_chars[:-1]

        return file_chars

try :
    files = ['eg1.txt', 'eg2.txt', 'eg3.txt']
    input = []
    for filename in files:
        input.append(read_file(filename))
    for sample in input:
        print(sample + '\n')
    bag_of_words(input)

except FileNotFoundError as fn:
    print(fn)
