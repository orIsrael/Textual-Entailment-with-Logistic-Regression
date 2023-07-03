from experiment_2 import Word2vecSNLIModel
from experiment_1 import SemanticModel
from experiment_3 import Doc2vecSNLIModel
from experiment_4 import DeepW2VSNLIModel

ARGS = {
    'dev': 'data/snli_1.0/snli_1.0_dev.jsonl',
    'train': 'data/snli_1.0/snli_1.0_train.jsonl',
    'test': 'data/snli_1.0/snli_1.0_test.jsonl',
}


CONFIG = {
    'train': ARGS['train'],
    'test': ARGS['test']
}


def test_model(m, train_file: str, test_file: str):
    for n_way in [2, 3]:
        model = m(n_way)
        model.train(train_file)
        model.test(test_file)
        print(str(n_way) + "-Way Accuracy:", model.accuracy)


def main(train_file: str, test_file: str):
    test_model(DeepW2VSNLIModel, train_file, test_file)
    # test_model(Doc2vecSNLIModel, train_file, test_file)
    # test_model(Word2vecSNLIModel, train_file, test_file)
    # test_model(SemanticModel, train_file, test_file)


if __name__ == '__main__':
    main(CONFIG['train'], CONFIG['test'])
