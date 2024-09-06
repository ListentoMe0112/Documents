import pickle
import numpy as np
from sklearn.metrics import accuracy_score

tags = ["nvoice", "first_transition", "normal_transition"]
models = {}

def build_sample(X1, X2, X3, X4):
    sample_X = []
    for i in range(len(X1)):
        for j in range(4):
            if j == 0:
                sample = X1[i]
                sample = sample.reshape(1, -1)
            if j == 1:
                sample = np.concatenate((X1[i], X2[i]))
                sample = sample.reshape(1, -1)
            if j == 2:
                sample = np.concatenate((X1[i],X3[i], X2[i]))
                sample = sample.reshape(1, -1)
            if j == 3:
                X2_tmp = X2[i].reshape(1,7)
                X3_tmp = X3[i].reshape(1,7)
                # print(X2_tmp.shape, X3_tmp.shape)
                # print(np.mean(np.concatenate((X2_tmp, X3_tmp), axis=0), axis=0).shape)
                sample = np.concatenate((X1[i], X4[i], np.mean(np.concatenate((X2_tmp, X3_tmp), axis=0), axis=0)))
                sample = sample.reshape(1,-1)
            print(sample.shape)
            sample_X.append(sample)
            
    return sample_X
            
def load_models():
    for tag in tags:
        with open('{}_scaler.pkl'.format(tag), "rb") as f:
            scaler = pickle.load(f)
        with open('{}_model.pkl'.format(tag), "rb") as f:
            model = pickle.load(f)

        models[tag] = {}
        models[tag]["scaler"] = scaler
        models[tag]["model"] = model
        

if __name__ == "__main__":
    load_models()
    X_test_1 = np.array([
        [1.001534882, 0.08794,  0.057143, 0.3,3.25,30.6,2.2],
        [1.507044905, 0.198444, 0.098039, 0.3, 2.333333333, 24.66666667,1.125],
        [0.787361222, 0.372093, 0.125,   0.3, 2.666666667, 15.4,1.333333333],
        [1.056952263, 0.094907, 0.121951, 0.3,2,0.444444444,1.428571429],
        [0.520742905, 0.168966, 0.122449, 0.3,5.666666667,16.44444444,1.714285714],
        [0.959293043, 0.12476,  0.076923, 0.3, 9.5,24.11111111,2.222222222],
        [0.940208882,0.16, 0.178571, 0.3,4.666666667,2.333333333,2],
        [0.940208882,0.16, 0.178571, 0.3,4.666666667,2.333333333,2],
        [0.764920758, 0.401163, 0.144928, 0.3, 3.428571429, 14.27272727, 1.727272727],
        [1.757389, 0.080645, 0.4, 0.3, 0.5, 3.25,1]
        ])
    
    X_test_2 = np.array([
        [0.942071045, 0.175573, 0.043478, 0.35, 2.833333333, 12.58333333,3],
        [1.082915167, 0.179878, 0.135593, 0.35, 2.571428571, 15.55555556,2],
        [0.789700871, 0.241573, 0.127907,0.35,2.5,12.08333333,1.5],
        [0.977073632,0.179402,0.037037,0.35,8.5,0.777777778,1.6],
        [0.802012333,0.208178,0.107143,0.35,2.285714286,0.9,2.090909091],
        [0.898833071,0.178771,0.15625,0.35,1.666666667,6.125,1.444444444],
        [0.711203929,0.270833,0.179487,0.35,2.6,5,1.857142857],
        [0.711203929,0.270833,0.179487,0.35,2.6,5,1.857142857],
        [0.678385,0.293333,0.045455,0.35,4.333333333,13.1,1.3],
        [1.052091667,0.120482,0.2,0.35,1.333333333,13.42857143,1.6],
        
    ]
    )
    
    X_test_3 = np.array([
        [0.731491222,0.105943,0.146341,0.4,0.769230769,17.61538462,1.928571429],
        [1.006819227,0.184848,0.196721,0.4,1.777777778,11.91666667,1.333333333],
        [0.535808154,0.350877,0.05,0.4,2.2,7.714285714,1.333333333],
        [0.802092187,0.148594,0.108108,0.4,1.5,0.222222222,2.428571429],
        [1.064942625,0.201271,0.105263,0.4,2.2,0.8,1.9375],
        [0.845595357,0.08589,0.119048,0.4,2.4,8.909090909,1.916666667],
        [0.782910759,0.199275,0.109091,0.4,2.833333333,3.076923077,1.875],
        [0.782910759,0.199275,0.109091,0.4,2.833333333,3.076923077,1.875],
        [0.726649727,0.394904,0.064516,0.4,2.125,12.83333333,1.272727273],
        [0.687894692,0.116608,0.090909,0.4,1.125,22.85714286,1.571428571],
    ])
    
    X_test_4 = np.array([
        [1.065873905,0.18232,0,0.35,0.7,1.272727273,1.636363636],
        [0.811695692,0.172205,0.087719,0.35,1.272727273,11.78571429,1.444444444],
        [0.861319389,0.228261,0.190476,0.35,1.5,6.888888889,1.5],
        [0.703776667,0.093103,0,0.35,0.888888889,1.111111111,2],
        [0.771748769,0.261628,0.088889,0.35,2.111111111,2.636363636,1.222222222],
        [0.605647667,0.185841,0.095238,0.35,1.375,1.5,1.416666667],
        [0.7433724,0.212644,0.189189,0.35,1.363636364,4.357142857,1.571428571],
        [0.7433724,0.212644,0.189189,0.35,1.363636364,4.357142857,1.571428571],
        [0.601535111,0.364238,0.036364,0.35,1.75,13.54545455,1.666666667],
        [0.7395452,0.304,0.157895,0.35,2,8.222222222,1.666666667],
    ])

    X_test = build_sample(X_test_1, X_test_2, X_test_3, X_test_4)    
    # with open('{}_x_test.pkl'.format(tag), "rb") as f:
    #     X_test = pickle.load(f)
    
    # with open('{}_y_test.pkl'.format(tag), "rb") as f:
    #     y_test = pickle.load(f)
        
    for single_sample in X_test:
        if single_sample.shape == (1,7):
            tag = "nvoice"
            single_sample = models[tag]["scaler"].transform(single_sample)
            y_pred = models[tag]["model"].predict(single_sample)
            decision_function_scores = models[tag]["model"].decision_function(single_sample)
            print(decision_function_scores)
        elif single_sample.shape == (1,14):
            tag = "first_transition"
            single_sample = models[tag]["scaler"].transform(single_sample)
            y_pred = models[tag]["model"].predict(single_sample)
            decision_function_scores = models[tag]["model"].decision_function(single_sample)
            print(decision_function_scores)
        elif single_sample.shape == (1,21):
            tag = "normal_transition"
            single_sample = models[tag]["scaler"].transform(single_sample)
            y_pred = models[tag]["model"].predict(single_sample)
            decision_function_scores = models[tag]["model"].decision_function(single_sample)
            print(decision_function_scores)
        print(y_pred)
    # X_test = scaler.transform(X_test)
    # y_pred = model.predict(X_test)
    # print(f'{tag} SVM Total Accuracy: {accuracy_score(y_test, y_pred):.2f}')