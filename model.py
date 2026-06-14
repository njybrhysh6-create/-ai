import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

data=pd.read_csv('data.csv')
# print(data.info())
# print(data.describe(include='all'))

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(0)

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    data[col] = data[col].astype('category').cat.codes

cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df_final = pd.get_dummies(data, columns=cat_cols, dtype=int)
X = df_final.drop(columns=['customerID', 'Churn'])
y = df_final['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 2. توليد البيانات الاصطناعية وموازنة بيانات التدريب فقط
# weights = compute_sample_weight('balanced', y_train) <- احذفها دامك تستخدم SMOTE

model = HistGradientBoostingClassifier(
    random_state=42,
    learning_rate=0.01, # <-- عدلته من 0.6
    max_iter=3000, # <-- عدلته من 400
    max_depth=None, # <-- عدلته من 14
    min_samples_leaf=20, # <-- عدلته من 10
     #class_weight='balanced', <- احذفها لأن SMOTE موازن الداتا خلاص
    early_stopping=True, # <-- أضفتها
    validation_fraction=0.1, # <-- أضفتها
    verbose=1
)
#model = RandomForestClassifier(n_estimators=600, max_depth=12, random_state=42, n_jobs=-1, verbose=2)




model.fit(X_train_balanced, y_train_balanced)

# 6. التقييم النهائي - هذا الجزء كله يتغير
y_proba = model.predict_proba(X_test)[:, 1] # <-- جيب الاحتمالات

# جرب thresholds عشان الفئة 1
print("\n--- تجربة Thresholds للفئة 1 ---")
for t in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    y_pred_t = (y_proba >= t).astype(int)
    rep = classification_report(y_test, y_pred_t, output_dict=True, zero_division=0)
    print(f"t={t:.2f} -> Recall1={rep['1']['recall']:.3f} | Precision1={rep['1']['precision']:.3f} | F1_1={rep['1']['f1-score']:.3f}")

# اختار أفضل threshold واطبع التقرير النهائي
best_t = 0.3# <-- غير الرقم حسب اللي طلع أحسن فوق
y_pred_final = (y_proba >= best_t).astype(int)

print(f"\n--- تقرير النتائج النهائي عند threshold={best_t} ---")
print('_____________________________')
print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final, digits=2, zero_division=0))