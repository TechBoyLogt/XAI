import pandas as pd
import dice_ml
from dice_ml.utils import helpers  # helper functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# 1. Load the dataset
df = pd.read_csv("diabetes.csv")
target = 'Outcome'
continuous_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 2. Split data and train a model
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Setup DiCE
# Create a DiCE data object
d = dice_ml.Data(dataframe=df, 
                 continuous_features=continuous_features, 
                 outcome_name=target)

# Create a DiCE model object
m = dice_ml.Model(model=clf, backend="sklearn")

# Initialize the explainer (using the 'random' method for speed/simplicity)
exp = dice_ml.Dice(d, m, method="random")

# 4. Generate Counterfactual Explanations
# Let's pick a patient from the test set who was predicted to have diabetes (Outcome=1)
query_instance = X_test.iloc[0:1] 

# Generate 4 diverse counterfactuals that show how the outcome could be 0
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_class=0)
# 5. Visualize the results
dice_exp.visualize_as_dataframe(show_only_changes=True)