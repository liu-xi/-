from train import test_accuracy
import pickle


if __name__ == '__main__':
    model_prameters_name = './Mnist_model.pkl'
    f = open(model_prameters_name, 'rb')
    param = pickle.load(f)
    # print(param)
    f.close

    accu = test_accuracy(param)
    print(accu)
