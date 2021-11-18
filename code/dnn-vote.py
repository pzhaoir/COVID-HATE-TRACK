import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import time


tic1 = time.perf_counter()
# Load data
data = pd.read_csv("main_data.csv")
train_data = data.iloc[:, 4: 13]

# retrieve numpy array
dataset = train_data.values
# split into input (X) and output (y) variables
X = dataset[:, :-1]
y = dataset[:,-1]
# format all fields as string
X = X.astype(str)
# reshape target to be a 2d array
y = y.reshape((len(y), 1))
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# define the  model
model = Sequential()
model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=2)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f ' % (accuracy*100))

# Performance evaluation
pred_class = model.predict_classes(X_test_enc)
pred_score = model.predict(X_test_enc)

F1_score = f1_score(y_test_enc, pred_class, average='macro')
print("F1 score is", F1_score)

accuracy_score = accuracy_score(y_test_enc, pred_class)
print("Acuracy is ", accuracy_score)

auc_score = roc_auc_score(y_test_enc, pred_score)
print("AUC score is ", auc_score)


# Plot ROC curve
false_positive_rate1_cnn, true_positive_rate1_cnn, threshold1_cnn = roc_curve(y_test_enc, pred_score)

plt.figure()
lw = 2
plt.plot(false_positive_rate1_cnn, true_positive_rate1_cnn, color='darkorange',
           label="AUC score = "+str(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for DNN')
plt.legend(loc="lower right")
plt.savefig('ROC_curve.png', dpi=300)
plt.show()

toc2 = time.perf_counter()
print(f"Total consumed running time: {toc2 - tic1: 0.4f} seconds.")
