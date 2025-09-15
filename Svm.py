import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,f1_score,classification_report,
    confusion_matrix,roc_auc_score,roc_curve
)

def run(show_plot=True,verbose=True,random_state=42):
    data=load_breast_cancer()
    x,y=data.data,data.target
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=random_state)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel='rbf',probability=True, random_state=random_state))
    ])

    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)
    y_prob=model.predict_proba(x_test)[:,1]
    
    metrics={
        "model":"Support Vector Machine",
        "accuracy":accuracy_score(y_test,y_pred),
        "f1":f1_score(y_test,y_pred),
        "acu":roc_auc_score(y_test,y_pred),
        "confusion_matrix":confusion_matrix(y_test,y_pred),
        "classification_report":classification_report(y_test,y_pred,digits=4)
    }
    
    if verbose:
        print("=== SVM ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 score: {metrics['f1']:.4f}")
        print(f"confusion matrix: \n{metrics['confusion_matrix']}")
        print(f"classification report: {metrics['classification_report']}")
        print(f"ROC & AUC: {metrics['acu']:.4f}")
        
    if show_plot:
        fpr,tpr,_=roc_curve(y_test,y_prob)
        plt.figure()
        plt.plot(fpr,tpr,label=f"AUC={metrics['acu']:.4f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("SR-ROC Curve")
        plt.legend()
        plt.show()
        
        return metrics
        
        
if __name__ == "__main__":
    run(show_plot=True, verbose=True)