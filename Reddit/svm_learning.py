from sklearn import svm
import bagging as bg

def process_files(files):
    input = []
    for filename in files:
        input.append(bg.read_file(filename))
    return input

def preprocess():
    '''
        Preprocess for training data
    :return: vectors i.e. dictionary for every sample read from a file
            with frequencies of the words
    '''
    try :
        files = ['eg1.txt', 'eg2.txt', 'eg3.txt']

        input = process_files(files)

        return bg.bag_of_words(input)

    except FileNotFoundError as fn:
        print(fn)

def process_for_prediction(vectors):
    '''
            Preprocess for testing data
        :return: vectors i.e. dictionary for every prediction sample read from a file
        with frequencies of the words used in training
        '''
    try :
        files = ['pred1']

        input = process_files(files)

        return bg.bag_predicted_data(vectors, input)

    except FileNotFoundError as fn:
        print(fn)

def svm_model():
    vectors = preprocess()

    X = []

    y = [0, 1, 0]

    for vector in vectors:
        X.append(list(vector.values()))

    #print(X)

    model = svm.SVC(gamma='scale')

    model.fit(X, y)

    predict_vector = process_for_prediction(vectors)

    print(predict_vector)

    predict_X = []

    for vector in predict_vector:
        predict_X.append(list(vector.values()))

    print(model.predict(predict_X))

svm_model()