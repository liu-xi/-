import pickle

if __name__ == '__main__':
    model_prameters_name = './best_model.pkl'
    f = open(model_prameters_name, 'rb')
    param = pickle.load(f)
    f.close

    print(f'layer0:{param[0]}')
    print(f'layer1:{param[1]}')