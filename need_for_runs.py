# import os
# import pickle
#
# with open(os.path.join('our_imdb', 'dict_class_measures_Node2Vec_cosine.pkl'), 'rb') as handle:
#     dict_class_measures = pickle.load(handle)
# keys = list(dict_class_measures.keys())
# count = 0
# for key in keys:
#     print(dict_class_measures[key]['acc'])
#     count += dict_class_measures[key]['acc']
# avg_acc = count/len(keys)
# print(avg_acc)

# from IMDb_data_preparation_E2V import MoviesGraph
# weights_dict = {'movies_edges': 0.6, 'labels_edges': 3.5}
# dict_paths = {'cast': 'data_set/IMDb title_principals.csv', 'genre': 'data_set/IMDb movies.csv'}
# IMDb = MoviesGraph(dict_paths)
# gnx = IMDb.create_graph()
# labels = IMDb.labels2int(gnx)
# print(labels.head())
# print(labels.shape)
# knowledge_gnx, knowledge_data = IMDb.create_knowledge_graph(labels)
# graph = IMDb.weighted_multi_graph(gnx, knowledge_gnx, labels, weights_dict)
# print('1')

from scipy.optimize import minimize
import numpy as np
import pandas as pd


def obj_func(x):
    try:
        values = pd.read_csv('our_imdb/train/optimaize_values_Node2Vec_l2.csv')
        result = pd.read_csv('our_imdb/train/optimaize_result_Node2Vec_l2.csv')
        df1 = pd.DataFrame(x.reshape(1, 3), columns=['a','b','c'])
        df2 = pd.DataFrame([x[0] + x[1] - x[2]], columns=['a'])
        frames1 = [values, df1]
        frames2 = [result, df2]
        values = pd.concat(frames1, axis=0, names=['a','b','c'])
        result = pd.concat(frames2, axis=0, names=['a'])
    except:
        values = pd.DataFrame(x.reshape(1, 3), columns=['a','b','c'])
        result = pd.DataFrame([x[0]+x[1]-x[2]], columns=['a'])
    values.to_csv('our_imdb/train/optimaize_values_Node2Vec_l2.csv', index=None)
    result.to_csv('our_imdb/train/optimaize_result_Node2Vec_l2.csv', index=None)
    return x[0]+x[1]-x[2]


x0 = np.array([1, 0.5, 0.2])
res = minimize(obj_func, x0=x0, method='Nelder-Mead')
print(res)

print(pd.read_csv('our_imdb/train/optimaize_values_Node2Vec_l2.csv'))
print(pd.read_csv('our_imdb/train/optimaize_result_Node2Vec_l2.csv'))