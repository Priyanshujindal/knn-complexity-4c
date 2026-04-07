from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt

data=load_iris()
print(data)
X=data.data
y=data.target

X_train ,X_test ,y_train ,y_test =train_test_split(
    X,y,test_size=0.3,random_state=42
)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

k_range=range(1,31)
cv_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k);
    # using cross-validation for stable accuracy estimate
    scores=cross_val_score(knn,X_train,y_train,cv=5);
    print(f"k={k},CV Scores: {scores}, Mean CV Score: {scores.mean()}")
    cv_scores.append(scores.mean())

# 3. Visualizing the 'Sweet Spot'
plt.figure(figsize=(8,5))
plt.plot(k_range,cv_scores,marker="*",color='green')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding optimal K (Bias-variance Tradeoff)')
plt.grid(True)
plt.show()