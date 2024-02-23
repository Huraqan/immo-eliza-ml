import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

feature_list = [
    "id",
    "price",
    "property_type",
    "subproperty_type",
    "region",
    "province",
    "locality",
    "zip_code",
    "latitude",
    "longitude",
    "construction_year",
    "total_area_sqm",
    "surface_land_sqm",
    "nbr_frontages",
    "nbr_bedrooms",
    "equipped_kitchen",
    "fl_furnished",
    "fl_open_fire",
    "fl_terrace",
    "terrace_sqm",
    "fl_garden",
    "garden_sqm",
    "fl_swimming_pool",
    "fl_floodzone",
    "state_building",
    "primary_energy_consumption_sqm",
    "epc",
    "heating_type",
    "fl_double_glazing",
    "cadastral_income",
]

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")
    
    # Define features to use
    num_features = [
        "total_area_sqm",
        "surface_land_sqm",
        "nbr_frontages",
        "primary_energy_consumption_sqm",
        "nbr_bedrooms",
        "terrace_sqm",
        "cadastral_income",
        "garden_sqm",
    ]
    fl_features = [
        "fl_terrace",
        "fl_garden",
        "fl_furnished",
        "fl_open_fire",
        "fl_double_glazing",
        "fl_swimming_pool",
        "fl_floodzone",
    ]
    cat_features = [
        "equipped_kitchen",
        "property_type",
        "subproperty_type",
        "region",
        "province",
        "locality",
        "epc",
        "heating_type",
        "state_building",
    ]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=69420
    )

    # Impute missing values using SimpleImputer
    # Many possible ways, see https://en.wikipedia.org/wiki/Imputation_(statistics)
    # SimpleImputer takes care of it but a specific strategy can be picked...
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])
    
    
    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()
    
    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    
    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    
    print(f"Features: \n {X_train.columns.tolist()}")
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")
    
    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()
