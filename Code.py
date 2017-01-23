import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

#1) Загрузка обучающих данных
news_train = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

iter = 1
news_train_data = []
for x in news_train:
    iter = iter + 1
    if iter != 4536:
        news_train_data.append(x[2])

iter = 1
news_train_data_target = []
for x in news_train:
    iter = iter + 1
    if iter != 4536:
        news_train_data_target.append(x[0])

#2) Перевод текста в числовой вектор признаков ("мешок слов")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(news_train_data)

#3) Предобработка текста, токенизация и отфильтровывание стоп-слов
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

#4) Вычисление частоты и обратной частоты термина (tf-idf)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#5) Обучение классификатора (naive_bayes, multinomial)
clf = MultinomialNB().fit(X_train_tfidf, news_train_data_target)
data_new = ['В Ираке новые танки']

#6) Извлечение характерных признаков (transform)
X_new_counts = count_vect.transform(data_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

#7) Создание конвейрной обработки (pipeline)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(news_train_data, news_train_data_target)

print("Receiving test data..")
test = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

print("Receiving test dataset..")
news_test_data = []
iter = 1
for x in test:
    iter = iter + 1
    if iter != 4536:
        news_test_data.append(x[2])

print("Receiving test dataset_target..")
iter = 1
news_test_data_target = []
for x in news_train:
    iter = iter + 1
    if iter != 4536:
        news_test_data_target.append(x[0])

print("Testing..")
docs_test = news_test_data

#8) Оценка производительности (predicted)
predicted = text_clf.predict(docs_test)
print (np.mean(predicted == news_test_data_target))

print("Receiving final dataset..")
news_test_final = list(csv.reader(open('news_test.txt', 'rt', encoding="utf8"), delimiter='\t'))

news_test_data_final = []
iter = 1
for x in news_test_final:
    iter = iter + 1
    if iter == 15002:
        news_test_data_final.append('Something went wrong...')
    if iter != 15002:
        news_test_data_final.append(x[1])
print (iter)

docs_test = news_test_data_final

#9) Применение линейного метода опорных векторов (подсоединение другого объекта классификации в готовый конвейер)
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(news_train_data, news_train_data_target)
predicted = text_clf.predict(docs_test)

#10) Вывод
print("Writing to file...")
fh = open("final_output.txt", 'w')
for item in predicted: fh.write("%s\n" % item)