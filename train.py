from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

tpot = TPOTClassifier(
    generations = 5,
    population_size = 10,
    verbosity = 2,
    random_state = 42,
    config_dict = 'TPOT sparse',
    memory = 'auto',
    n_jobs = -1,
    cv = 5
)

tpot.fit(X_train, y_train)

accuracy = tpot.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

tpot.export('best_model_pipeline.py')
