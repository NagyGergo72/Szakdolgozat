###################################################################################
#Használt könyvtárak importálása
###################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pyodbc
import yaml
from pingouin import partial_corr
import scipy


###################################################################################
#Adatok betöltése Azure szerveren tárolt adatbázisból
###################################################################################
def activate_database_driver(driver_version="18",credentials_file='credentials_gergo.yml'):

    with open(credentials_file,'r') as yaml_file:
        credentials=yaml.safe_load(yaml_file)

    db_database = credentials['database']['database']
    db_server = credentials['database']['server']
    db_user = credentials['database']['username']
    db_password = credentials['database']['password']
    pyodbc.drivers()
    conn = pyodbc.connect(
        "Driver={ODBC Driver "+driver_version+" for SQL Server};"
        "Server=tcp:"+db_server+";"
        "Port=1433;"
        "Database="+db_database+";"
        "Encrypt=yes;"
        "Uid="+db_user+";"
        "Pwd={"+db_password+"};"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    return conn

query="select * from dbo.Final$"
df = pd.read_sql(query,activate_database_driver(driver_version="18",credentials_file='credentials_gergo.yml'))

df = df.iloc[:, 1:] 
df.set_index(df.columns[0], inplace=True) 
df.columns = pd.to_datetime(df.columns, format='%Y')
df = df.astype(float)
df = df.interpolate(method='cubicspline', axis=1).bfill(axis=1)
df.T.dtypes


###################################################################################
#Adatok betöltése lokális Excelből
###################################################################################
df = pd.read_excel('Adatok/Adatok Modell/Modell_Migráció.xlsx', header=0, index_col=2, sheet_name='Final')
df = df.iloc[:, 2:]
df.columns = pd.to_datetime(df.columns, format='%Y')
df = df.astype(float)
df = df.interpolate(method='cubicspline', axis=1).bfill(axis=1)
df.T.dtypes


###################################################################################
#Adatok kezdetleges vizsgálata vizualizációval
###################################################################################
df_2 = pd.read_excel('Adatok/Model_Népesség_Final.xlsx', header=0, index_col=1, sheet_name='Final')
df_2 = df_2.drop(columns=["Változók rövidítve", "Forrás"], errors='ignore')
df_2.columns = pd.to_datetime(df_2.columns, format='%Y')
df_2 = df_2.replace(',', '.', regex=True)
df_2 = df_2.astype(float)
df_2 = df_2.interpolate(method='linear', axis=1).bfill(axis=1)

#A4-es lapmérehez igazitás
fig, axes = plt.subplots(nrows=(len(df_2) + 1) // 2, ncols=2, figsize=(8.27, 11.69), constrained_layout=True)


axes = axes.flatten()
# Tényezők idősorainak vizualizálása, soronként 2
for i, (index, row_data) in enumerate(df_2.iterrows()):
    ax = axes[i]
    ax.plot(df_2.columns, row_data.values, linestyle='-', label=index)
    ax.scatter(df_2.columns, row_data.values, color='red', label='Értékek')
    title_words = index.split()
    # Ha cím túl hosszú, két sorban jelenítjük meg
    if len(" ".join(title_words)) <= 30:
        ax.set_title(" ".join(title_words), fontsize=10)
    else:
        ax.set_title("\n".join([" ".join(title_words[:3]), " ".join(title_words[3:])]), fontsize=10)
    ax.set_xlabel("Év")

# Nem használt tengelyek eltávolítása
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()




###################################################################################
#Up-sampling és interpoláció
###################################################################################
index_original = pd.date_range(start="2001", periods=df.shape[1], freq="A")  # Éves felbontás
index_target = pd.date_range("2001-01-01", "2023-12-31", freq="M")  # Havi felbontás

upsampled_data = pd.DataFrame(index=index_target)

for elem, series in df.iterrows():
    if not series.isna().all():
        # Üres index létrehozása a cél idősorhoz és az eredeti idősor értékeinek másolása a cél idősorba
        interpol = pd.Series(index=index_target, dtype="float64")
        interpol.loc[index_original] = series.values
        
        valid_index = index_original[~series.isna()]
        valid_values = series.dropna().values
        
        dist_mean, dist_std = norm.fit(valid_values)  # Normális eloszlás becslése
        
        # Hiányzó értékek kitöltése az illesztett eloszlással
        missing_index = interpol[interpol.isnull()].index
        simulated_values = norm.rvs(loc=dist_mean, scale=dist_std, size=len(missing_index), random_state=72)
        interpol.loc[missing_index] = simulated_values
        
        # Hozzáadás az upsampled_data DataFrame-hez
        upsampled_data[elem] = interpol

df_upsampled = pd.DataFrame(upsampled_data)





###################################################################################
#Normalizálás eredményének vizsgálata különböző scalerek alkalmazásával
###################################################################################
scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
} # Skálázási módszerek

for scaler_name, scaler in scalers.items():
    normalized_data = pd.DataFrame(
        scaler.fit_transform(df_upsampled.T),
        index=df_upsampled.T.index,
        columns=df_upsampled.T.columns
    )

    # Skálázott idősorok vizualizálása 
    fig, axes = plt.subplots(5, 4, figsize=(30, 15), sharex=False)
    axes = axes.flatten()

    for i, (index, row) in enumerate(normalized_data.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.plot(normalized_data.columns, row.values, linestyle='-', label=index)
        ax.set_title(f"{index}")
        ax.set_xlabel("Év")
        ax.set_ylabel("Normalizált értékek")
        ax.grid(True)
        ax.legend()

    # Nem használt tengelyek eltávolítása
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Normalizált változó, skálázás típusa: {scaler_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


###################################################################################
#Normalizálás RobustScaler alkalmazásával
###################################################################################
scaler = RobustScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(df_upsampled),
    index=df_upsampled.index,
    columns=df_upsampled.columns
)




###################################################################################
#Multikollinearitás vizsgálata
###################################################################################
def calculate_vif_with_reduction(df, threshold):
    while True:
        # VIF értékek számítása minden tényezőre
        vif_values = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif_df = pd.DataFrame({'Változó': df.columns, 'VIF': vif_values})
        
        # Legnagyobb VIF értékkel rendelkező változó azonosítása
        max_vif = vif_df['VIF'].max()
        
        # Ha a megadott küszöbérték alatt van a legnagyobb VIF, akkor kilépünk
        if max_vif < threshold:
            break
        
        # Legnagyobb VIF értékű változó eltávolítása
        max_vif_var = vif_df.loc[vif_df['VIF'].idxmax(), 'Változó']
        print(f"Tényező eltávolítása '{max_vif_var}' VIF érték = {max_vif:.2f}")
        df = df.drop(columns=[max_vif_var])
    
    return vif_df, df


#VIF számítás és változók kiejtése
df_cel = normalized_data.T[:1].T
vif_df, reduced_features = calculate_vif_with_reduction(normalized_data.T[1:].T, threshold=40.0)
combined_df = pd.concat([df_cel, reduced_features], axis=1)
df = combined_df





###################################################################################
#Stacionaritás vizsgálata ADF teszttel
###################################################################################
def check_stationarity(df):
    p_values = {}
    for column in df.columns:
        result = adfuller(df[column].dropna()) # ADF teszt alkalmazása
        p_values[column] = result[1]
        print(f"{column} - ADF Stat: {result[0]:.2f}, p-érték: {result[1]:}") # p-értékek kiírása
    return p_values # p-értékek kimentése

p_values = check_stationarity(df)




###################################################################################
#Differenciálás alkalmazása amennyiben az ADF teszt p-értéke > 0.05
###################################################################################
df_upsampled_stacionary = df.T.copy()
alpha = 0.05
# Differenciálás alkalmazása a nem stacioner idősorokra
for index, p_value in p_values.items():
    differenced_series = df_upsampled_stacionary.loc[index]
    diff_count = 0
    while p_value > alpha: #Többszörtös differenciálás alkalmazása, amíg a p-érték > 0.05
        differenced_series = differenced_series.diff().dropna()
        result = adfuller(differenced_series)
        p_value = result[1]
        diff_count += 1
        print(f"{index} - Differenciálási szint {diff_count}: ADF Stat: {result[0]:.2f}, p-érték: {p_value:}")
    df_upsampled_stacionary.loc[index] = differenced_series

#Üres értékek eltávolítása
df_upsampled_stacionary.dropna(how='all', inplace=True)

# Lineáris interpoláció alkalmazása a hiányzó értékek kitöltésére
df_upsampled_stacionary.interpolate(method='linear', inplace=True)
df_upsampled_stacionary.fillna(method='ffill', inplace=True)  
df_upsampled_stacionary.fillna(method='bfill', inplace=True) 




###################################################################################
#Stacioner idősorok vizuális vizsgálata
###################################################################################
fig, axes = plt.subplots(5, 4, figsize=(30, 15), sharex=False)
axes = axes.flatten()
for i, (index, row) in enumerate(df.T.iterrows()):
    if i >= len(axes):
        break
    ax = axes[i]
    ax.plot(df.T.columns, row.values, linestyle='-', label=index)
    ax.set_title(f"{index}")
    ax.set_xlabel("Év")
    ax.set_ylabel("Érték")
    ax.grid(True)
    ax.legend()
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()




########################################################################################
#Train és test adatok szétválasztása, változók szétbontása cél- és magyarázó változókra
########################################################################################
df = df.copy()

target_variable = 'Cél'

#Tényezők szétbontása cél- és magyarázó változókra
X = df.drop(columns=[target_variable])  # Magyarázó változók
y = df[target_variable]  # Célváltozó

# Tanító és teszt adatokra bontás 80% tanító, 20% teszt
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Magyarázó változók kiválasztása
feature_columns = [col for col in df.columns if col in X_train.columns]

Random_State = 72
min_features = 4 # Minimális korlát változószelekcióban a változók számának
max_features = len(feature_columns)

# Összes változókombináció száma
total_combinations = sum(len(list(combinations(feature_columns, feature_count))) 
                         for feature_count in range(min_features, max_features + 1))


                  
########################################################################################
#Sarimax model futtatása
########################################################################################
param_grid_arimax = {
    'order': [(p, 1, q) for p in range(0, 3) for q in range(0, 3)],
    'seasonal_order': [(P, 1, Q, 12) for P in range(0, 3) for Q in range(0, 3)]
} # SARIMAX hiperparaméter rács

best_arimax = [] # Legjobb modell, ide mentjük a legjobb eredményeket: rmse, r2 és változók
best_arimax_partial_coeffs = [] # Legjobb modell parciális korrelációs együtthatói
best_summary_arimax = [] # Legjobb modell összegzése


def run_arimax():
    best_arimax_rmse = float('inf')
    best_arimax_r2 = None
    best_arimax_features = None
    best_summary = None

    for feature_count in range(min_features, max_features + 1):
        for feature_combination in combinations(feature_columns, feature_count):
            X_train_subset = X_train[list(feature_combination)]
            X_test_subset = X_test[list(feature_combination)]
            # Hiperparaméterek rácsának iterálása, modell létrehozása
            for order in param_grid_arimax['order']:
                for seasonal_order in param_grid_arimax['seasonal_order']:
                    arimax_model = SARIMAX(
                        y_train,
                        exog=X_train_subset,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    # Modell illesztése, vizsgált mutatók számítása
                    arimax_fit = arimax_model.fit(disp=False)
                    arimax_forecast = arimax_fit.get_forecast(steps=len(y_test), exog=X_test_subset).predicted_mean
                    arimax_rmse = np.sqrt(mean_squared_error(y_test, arimax_forecast))
                    arimax_r2 = r2_score(y_test, arimax_forecast)

                    if arimax_rmse < best_arimax_rmse:
                        #Legjobb modell vizsgált mutatóinak és változókombinációjának mentése
                        best_arimax_features = feature_combination
                        best_arimax_rmse = arimax_rmse
                        best_arimax_r2 = arimax_r2
                        best_summary = arimax_fit.summary()

                        # Legjobb modell összegzésének mentése
                        best_summary_arimax.clear()
                        best_summary_arimax.append(best_summary)

                        # Parciális korrelációs együtthatók számítása
                        partial_corrs = {}
                        for i, feature in enumerate(feature_combination):
                            partial_corr_result = partial_corr(
                                data=pd.concat([X_train_subset, y_train], axis=1),
                                x=feature,
                                y=y_train.name,
                                covar=[col for col in feature_combination if col != feature]
                            )
                            partial_corrs[feature] = partial_corr_result['r'].values[0]
                        best_arimax_partial_coeffs.clear()
                        best_arimax_partial_coeffs.append(partial_corrs)


    best_arimax.append({'Változók': best_arimax_features, 'Modell': 'SARIMAX', 'RMSE': best_arimax_rmse, 'R²': best_arimax_r2})

run_arimax()

# Összegző adattábla létrehozása a legjobb eredményekből
summary_data_arimax = []

summary_data_arimax.append({
    'Modell': "SARIMAX",
    'R²': best_arimax[0]['R²'],
    'RMSE': best_arimax[0]['RMSE'],
    'Parciális korrelációs együtthatók': best_arimax_partial_coeffs,
    'Változók': best_arimax[0]['Változók'],
    'Összegző tábla': best_summary_arimax,
})

summary_df_arimax = pd.DataFrame(summary_data_arimax)

# P-érték kalkulálása SARIMAX modell összegzéséből (kétoldalú z-próba)
def calculate_p_values_twosided(z_score):
    p_value = scipy.stats.norm.sf(abs(z_score))*2
    return p_value

calculate_p_values_twosided()




########################################################################################
#Ridge modell futtatása
########################################################################################
Ridge_Parameters = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['auto', 'sag', 'lsqr', 'saga'],
    'tol': [1e-4, 1e-3, 1e-2],
    'max_iter': [1000, 5000, 10000, 50000],
} # Ridge hiperparaméter rács


best_ridge = [] # Legjobb modell, ide mentjük a legjobb eredményeket: rmse, r2 és változók
best_ridge_weights = [] # Legjobb modell súlyai
best_ridge_partial_coeffs = [] # Legjobb modell parciális korrelációs együtthatói
best_ridge_significance = [] # Legjobb modell tényezőinek szignifikanciája (p-értékek)


def run_ridge():
    best_ridge_rmse = float('inf')
    best_ridge_r2 = None
    best_ridge_features = None

    # Változó kombináció kiválasztása és modell illesztése
    for feature_count in range(min_features, max_features + 1):
        for feature_combination in combinations(feature_columns, feature_count):
            X_train_subset = X_train[list(feature_combination)]
            X_test_subset = X_test[list(feature_combination)]

            ridge_model = Ridge(random_state=Random_State)
            ridge_grid_search = GridSearchCV(ridge_model, Ridge_Parameters, cv=3, n_jobs=-1, verbose=0) # Optimális hiperparaméterek keresése
            ridge_grid_search.fit(X_train_subset, y_train)
            current_ridge_model = ridge_grid_search.best_estimator_
            ridge_forecast = current_ridge_model.predict(X_test_subset)
            ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_forecast))
            ridge_r2 = r2_score(y_test, ridge_forecast)

            # Legjobb modell kiválasztása és mutatók mentése
            if ridge_rmse < best_ridge_rmse:
                best_ridge_rmse = ridge_rmse
                best_ridge_r2 = ridge_r2
                best_ridge_features = feature_combination
                

                # Súlyok mentése legjobb modellhez
                best_ridge_weights.clear()
                total_weight = sum(abs(current_ridge_model.coef_))
                normalized_weights = [(abs(coef) / total_weight) * 100 for coef in current_ridge_model.coef_]
                best_ridge_weights.extend(normalized_weights)

                # Parciális korrelációs együtthatók számítása legjobb modellhez
                partial_corrs = {}
                for i, feature in enumerate(feature_combination):
                    partial_corrs[feature] = np.corrcoef(
                        X_train_subset[feature], y_train - X_train_subset.drop(columns=[feature]).dot(current_ridge_model.coef_[np.arange(len(current_ridge_model.coef_)) != i])
                    )[0, 1]
                best_ridge_partial_coeffs.clear()
                best_ridge_partial_coeffs.append(partial_corrs)

                # Szignifikancia (p-értékek) számítása legjobb modellhez
                significance = {}
                for i, feature in enumerate(feature_combination):
                    t_stat = current_ridge_model.coef_[i] / (np.std(y_train) / np.sqrt(len(y_train)))
                    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
                    significance[feature] = f"{p_value:}"
                best_ridge_significance.clear()
                best_ridge_significance.append(significance)

    # Eredmények mentése legjobb modellhez
    best_ridge.append({'Változók': best_ridge_features, 'Modell': 'Ridge', 'RMSE': best_ridge_rmse, 'R²': best_ridge_r2})

run_ridge()

# Összegző adattábla létrehozása a legjobb eredményekből
summary_data_ridge = []

summary_data_ridge.append({
    'Modell': "Ridge",
    'R²': best_ridge[0]['R²'],
    'RMSE': best_ridge[0]['RMSE'],
    'Súlyok': best_ridge_weights,
    'Parciális korrelációs együtthatók': best_ridge_partial_coeffs,
    'Szignifikancia (p-érték)': best_ridge_significance,
    'Változók': best_ridge[0]['Változók']
})

summary_df_ridge = pd.DataFrame(summary_data_ridge)




########################################################################################
#Lasso modell futtatása
########################################################################################
Lasso_Parameters = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [1000, 5000, 10000, 50000],
    'tol': [1e-4, 1e-3, 1e-2],
    'selection': ['cyclic', 'random'],
} # Lasso hiperparaméter rács

best_lasso = [] # Legjobb modell, ide mentjük a legjobb eredményeket: rmse, r2 és változók
best_lasso_weights = [] # Legjobb modell súlyai
best_lasso_partial_coeffs = [] # Legjobb modell parciális korrelációs együtthatói
best_lasso_significance = [] # Legjobb modell tényezőinek szignifikanciája (p-értékek)


def run_lasso():
    best_lasso_rmse = float('inf')
    best_lasso_r2 = None
    best_lasso_features = None

    # Változó kombináció kiválasztása és modell illesztése
    for feature_count in range(min_features, max_features + 1):
        for feature_combination in combinations(feature_columns, feature_count):
            X_train_subset = X_train[list(feature_combination)]
            X_test_subset = X_test[list(feature_combination)]

            lasso_model = Lasso(random_state=Random_State)
            lasso_grid_search = GridSearchCV(lasso_model, Lasso_Parameters, cv=3, n_jobs=-1, verbose=0) # Optimális hiperparaméterek keresése
            lasso_grid_search.fit(X_train_subset, y_train)
            current_lasso_model = lasso_grid_search.best_estimator_
            lasso_forecast = current_lasso_model.predict(X_test_subset)
            lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_forecast))
            lasso_r2 = r2_score(y_test, lasso_forecast)

            # Legjobb modell kiválasztása és mutatók mentése
            if lasso_rmse < best_lasso_rmse:
                best_lasso_rmse = lasso_rmse
                best_lasso_r2 = lasso_r2
                best_lasso_features = feature_combination

                # Súlyok mentése legjobb modellhez
                best_lasso_weights.clear()
                total_weight = sum(abs(current_lasso_model.coef_))
                normalized_weights = [(abs(coef) / total_weight) * 100 for coef in current_lasso_model.coef_]
                best_lasso_weights.extend(normalized_weights)

                # Parciális korrelációs együtthatók számítása legjobb modellhez
                partial_corrs = {}
                for i, feature in enumerate(feature_combination):
                    partial_corrs[feature] = np.corrcoef(
                        X_train_subset[feature], y_train - X_train_subset.drop(columns=[feature]).dot(current_lasso_model.coef_[np.arange(len(current_lasso_model.coef_)) != i])
                    )[0, 1]
                best_lasso_partial_coeffs.clear()
                best_lasso_partial_coeffs.append(partial_corrs)


                # Szignifikancia (p-értékek) számítása legjobb modellhez
                significance = {}
                for i, feature in enumerate(feature_combination):
                    t_stat = current_lasso_model.coef_[i] / (np.std(y_train) / np.sqrt(len(y_train)))
                    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
                    significance[feature] = f"{p_value}" 
                best_lasso_significance.clear()
                best_lasso_significance.append(significance)

    # Eredmények mentése legjobb modellhez
    best_lasso.append({'Változók': best_lasso_features, 'Modell': 'Lasso', 'RMSE': best_lasso_rmse, 'R²': best_lasso_r2})


run_lasso()

# Összegző adattábla létrehozása a legjobb eredményekből
summary_data_lasso = []

summary_data_lasso.append({
    'Modell': "Lasso",
    'R²': best_lasso[0]['R²'],
    'RMSE': best_lasso[0]['RMSE'],
    'Súlyok': best_lasso_weights,
    'Parciális korrelációs együtthatók': best_lasso_partial_coeffs,
    'Szignifikancia (p-érték)': best_lasso_significance,
    'Változók': best_lasso[0]['Változók']
})

summary_df_lasso = pd.DataFrame(summary_data_lasso)




########################################################################################
#XGBoost modell futtatása
########################################################################################
XGBoost_Parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.7, 0.8],
} # XGBoost hiperparaméter rács

best_xgboost = [] # Legjobb modell, ide mentjük a legjobb eredményeket: rmse, r2 és változók
best_xgb_weights = [] # Legjobb modell súlyai, változók fontossági sorrendjének értékeléséhez

def run_xgboost():
    best_xgb_rmse = float('inf')
    best_xgb_r2 = None
    best_xgb_features = None

    # Változó kombináció kiválasztása és modell illesztése
    for feature_count in range(min_features, max_features + 1):
        for feature_combination in combinations(feature_columns, feature_count):
            X_train_subset = X_train[list(feature_combination)]
            X_test_subset = X_test[list(feature_combination)]

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=Random_State)
            xgb_grid_search = GridSearchCV(xgb_model, XGBoost_Parameters, cv=3, n_jobs=-1, verbose=0) # Optimális hiperparaméterek keresése
            xgb_grid_search.fit(X_train_subset, y_train)
            current_xgb_model = xgb_grid_search.best_estimator_
            xgb_forecast = current_xgb_model.predict(X_test_subset)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_forecast))
            xgb_r2 = r2_score(y_test, xgb_forecast)

            # Legjobb modell kiválasztása és mutatók mentése
            if xgb_rmse < best_xgb_rmse:
                best_xgb_rmse = xgb_rmse
                best_xgb_r2 = xgb_r2
                best_xgb_features = feature_combination

                # Változók fontosságának mentése súlyozásuk alapján
                best_xgb_weights.clear()
                total_importance = sum(current_xgb_model.feature_importances_)
                normalized_importances = [(importance / total_importance) * 100 for importance in current_xgb_model.feature_importances_]
                best_xgb_weights.extend(normalized_importances)

    best_xgboost.append({'Változók': best_xgb_features, 'Modell': 'XGBoost', 'RMSE': best_xgb_rmse, 'R²': best_xgb_r2})

run_xgboost()

# Összegző adattábla létrehozása a legjobb eredményekből
summary_data_xgb = []

summary_data_xgb.append({
    'Modell': "XGBoost",
    'R²': best_xgboost[0]['R²'],
    'RMSE': best_xgboost[0]['RMSE'],
    'Súlyok': best_xgb_weights,
    'Változók': best_xgboost[0]['Változók']
})

summary_df_xgb = pd.DataFrame(summary_data_xgb)


########################################################################################
#Eredmények kiiratása Excel fájlba
########################################################################################
with pd.ExcelWriter('Adatok/Model_Migráció_Eredmények.xlsx', engine='openpyxl') as writer:
    summary_df_ridge.to_excel(writer, sheet_name='Ridge', index=False)
    summary_df_lasso.to_excel(writer, sheet_name='Lasso', index=False)
    summary_df_xgb.to_excel(writer, sheet_name='XGBoost', index=False)
    summary_df_arimax.to_excel(writer, sheet_name='SARIMAX', index=False)