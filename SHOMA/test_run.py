import json
from script import Script

with open('langs.json') as f:
    langs = json.load(f)

languages = ['ES', 'EN'] 

for lang in languages:
    l = Script(lang, 
                langs[lang]['train'],
                langs[lang]['dev'],
                langs[lang]['test'],
                langs[lang]['word2vec_dir'],
                langs[lang]['model_name'],
                langs[lang]['initial_weight'])
    # parames: epoch, batch_size, pos
    l.set_params(10, 100, True) # here you can change epoch and batch. Default is 100 for both
    l.read_train_test()
    l.train_predict_test()
    del l


