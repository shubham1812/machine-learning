# machine-learning
Machine learning Algorithms

Implemented machine learning algorithms:
1) KNN
2) Adaboost
3) Random Forest - A faster implementation

To run : ./orient.py test test_file.txt model_file.txt [model]

Example: For Training

./orient.py train train-data.txt nearest_model.txt nearest ./orient.py train train-data.txt adaboost_model.txt adaboost ./orient.py train train-data.txt forest_model.txt forest ./orient.py train train-data.txt best_model.txt best

For Testing

./orient.py test test-data.txt nearest_model.txt nearest ./orient.py test test-data.txt adaboost_model.txt adaboost ./orient.py test test-data.txt forest_model.txt forest ./orient.py test test-data.txt best_model.txt best
