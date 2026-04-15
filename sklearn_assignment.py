from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Task - 1
def model_using_standard_scaler(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Accuracy: Train =", model.score(X_train, y_train).__round__(2)," Test = ", model.score(X_test, y_test).__round__(2))

print("Task - 1\n")
model_using_standard_scaler(X, y)
'''
Here in task 1 the test data is given along with train data for Standardscaler transformation. Since the model have access to test data 
which it shouldn't causing data leakage.
'''

# Task - 2
def model_using_pipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')

    print(scores)
    print(scores.mean().__round__(2))
    print(scores.std().__round__(2))
print("\nTask - 2\n")
model_using_pipeline(X, y)

# Task - 3
def model_using_decision_tree_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    for depth in [1, 5, 20]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)

        model.fit(X_train, y_train)
        print("Depth: ", depth)
        print("Train Accuracy: ", model.score(X_train, y_train).__round__(2))
        print("Test Accuracy: ", model.score(X_test, y_test).__round__(2))

print("\nTask - 3\n")
model_using_decision_tree_classifier(X, y)

'''
Depth 5 has balanced accuracy since the training accuracy is decent and not hundered percentage and 
gap between train and test accuracy is smaller compared to depth 20.
'''