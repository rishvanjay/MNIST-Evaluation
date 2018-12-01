import numpy as np
from matplotlib import pyplot
import final


X_train, X_test, y_train, y_test = final.getdata()

# resEp1 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.5)
# resEp5 = final.run(X_train, X_test, y_train, y_test, 5, 0.01, 0.5)
# resEp10 = final.run(X_train, X_test, y_train, y_test, 10, 0.01, 0.5)
# resEp100 = final.run(X_train, X_test, y_train, y_test, 100, 0.01, 0.5)


# resPen001 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.01)
# resPen01 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.1)
# resPen1 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 1)
# resPen10 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 10)


# resGam001 = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.01)
# resGam01 = final.run(X_train, X_test, y_train, y_test, 1, 0.1, 0.01)
# resGam1 = final.run(X_train, X_test, y_train, y_test, 1, 1, 0.01)

# resCompDigits = final.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.5)

# accuracy vs hyperparameter graphs


#comparing number of epochs with overall accuracy

# res1 = np.mean(resEp100[0])
# res5 = np.mean(resEp100[4])
# res10 = np.mean(resEp100[9])
# res50 = np.mean(resEp100[49])
# res100 = np.mean(resEp100[99])
# y = [res1, res5, res10, res50, res100]
# x = [1, 5, 10, 50, 100]
# y = []
# x = []
# for i, l in enumerate(resEp100):
#     y.append(np.mean(resEp100[i]))
#     x.append(i+1)

# pyplot.plot(x, y)
# pyplot.title('Number of Epochs vs Overall Accuracy')
# pyplot.xlabel('Number of Epochs')
# pyplot.ylabel('Accuracy')
# pyplot.ylim(0.94, 0.98)
# pyplot.show()


#comparing overall accuracy with different penalty  

# resPen001 = np.mean(resPen001)
# resPen01 = np.mean(resPen01)
# resPen1 = np.mean(resPen1)
# resPen10 = np.mean(resPen10)

# y = [resPen001, resPen01, resPen1, resPen10]
# x = [0.01, 0.1, 1, 10]

# y = []
# x = []
# penalties = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for p in penalties:

#     y.append(np.mean(final.run(X_train, X_test, y_train, y_test, 1, 0.01, p)))
#     x.append(p)



# pyplot.plot(x, y)
# pyplot.title('Penalty vs Overall Accuracy')
# pyplot.xlabel('Penalty')
# pyplot.ylabel('Accuracy')
# pyplot.show()

#comparing overall accuracy with different gamma

# resGam001 = np.mean(resGam001)
# resGam01 = np.mean(resGam01)
# resGam1 = np.mean(resGam1)
# y = [resGam001, resGam01, resGam1]
# x = [0.01, 0.1, 1]
# pyplot.plot(x, y)
# pyplot.title('Gamma vs Overall Accuracy')
# pyplot.xlabel('Gamma')
# pyplot.ylabel('Accuracy')
# pyplot.show()


#compare which digits train more accurately

# y = resCompDigits[0]
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# pyplot.plot(x, y)
# pyplot.title('Digit vs Training Accuracy')
# pyplot.xlabel('Digit')
# pyplot.ylabel('Accuracy')
# pyplot.show()



