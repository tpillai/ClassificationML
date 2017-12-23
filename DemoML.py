from sklearn import tree

classifier = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...


# [height, weight, shoe_size]
X = [[182, 80, 44], [176, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#  train them on our data
updatedClassifier = classifier.fit(X, Y)

myprediction = updatedClassifier.predict([[192, 70, 43]])

# compare their results and print the best one!

print(myprediction)