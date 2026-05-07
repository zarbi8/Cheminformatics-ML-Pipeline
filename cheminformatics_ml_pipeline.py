import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# RDKit imports for Cheminformatics
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
except ImportError:
    raise ImportError("RDKit is not installed. Please install it using: pip install rdkit")

warnings.filterwarnings('ignore')

class CheminformaticsMLSystem:
    def __init__(self, target_properties=['HOMO', 'LUMO', 'Bandgap']):
        self.target_properties = target_properties
        self.models = {} 
        self.scalers = {} 
        self.features_list = [
            'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
            'NumAromaticRings', 'NumRotatableBonds', 'FormalCharge', 'HeavyAtomCount'
        ]
        # Adding 2048 Morgan Fingerprint columns
        self.fp_cols = [f'FP_{i}' for i in range(2048)]
        self.all_feature_names = self.features_list + self.fp_cols

    def validate_and_standardize_smiles(self, smiles_list):
        valid_smiles = []
        valid_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                valid_smiles.append(canonical_smi)
                valid_mols.append(mol)
            else:
                print(f"Warning: Invalid SMILES skipped -> {smi}")
        return valid_smiles, valid_mols

    def generate_features(self, smiles_list):
        valid_smiles, valid_mols = self.validate_and_standardize_smiles(smiles_list)
        features = []
        
        for mol in valid_mols:
            desc = {
                'MolWt': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'FormalCharge': Chem.GetFormalCharge(mol),
                'HeavyAtomCount': mol.GetNumHeavyAtoms()
            }
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.zeros((0,), dtype=np.int8)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, fp_array)
            
            for i, bit in enumerate(fp_array):
                desc[f'FP_{i}'] = bit
                
            features.append(desc)
            
        df_features = pd.DataFrame(features)
        df_features.insert(0, 'SMILES', valid_smiles)
        return df_features

    def train_models(self, df_data):
        print("\n--- Training Models ---")
        df_clean = df_data.dropna() 
        X = df_clean[self.all_feature_names]
        
        for target in self.target_properties:
            if target not in df_clean.columns: continue
            y = df_clean[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target] = scaler
            self.models[target] = {}
            
            print(f"Training models for target: {target}...")
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            self.models[target]['RF'] = rf
            
            xgb = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
            xgb.fit(X_train_scaled, y_train)
            self.models[target]['XGB'] = xgb
            
            self._evaluate_model(target, 'RF', rf, X_test_scaled, y_test)
            self._evaluate_model(target, 'XGB', xgb, X_test_scaled, y_test)

    def _evaluate_model(self, target, model_name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  [{model_name}] R2: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

    def virtual_screening(self, smiles_list, rank_by=None, ascending=False):
        print("\n--- Virtual Screening ---")
        df_features = self.generate_features(smiles_list)
        if df_features.empty: return pd.DataFrame()
            
        X_new = df_features[self.all_feature_names]
        results = df_features[['SMILES']].copy()
        
        for target in self.target_properties:
            if target in self.models:
                scaler = self.scalers[target]
                X_new_scaled = scaler.transform(X_new)
                best_model = self.models[target]['XGB'] 
                results[f'Predicted_{target}'] = best_model.predict(X_new_scaled)
                
        if rank_by and f'Predicted_{rank_by}' in results.columns:
            results = results.sort_values(by=f'Predicted_{rank_by}', ascending=ascending).reset_index(drop=True)
        return results

    def generate_visualizations(self, target, y_true, y_pred, model_name, model_obj):
        print(f"\n--- Generating Visualizations for {target} ---")
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        plt.title(f'{target}: Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True)
        plt.title('Residual Error Distribution')
        plt.xlabel('Error')
        
        plt.subplot(1, 3, 3)
        importances = model_obj.feature_importances_
        desc_importances = importances[:len(self.features_list)]
        indices = np.argsort(desc_importances)[::-1]
        
        top_features = [self.features_list[i] for i in indices[:10]]
        top_imps = desc_importances[indices[:10]]
        
        sns.barplot(x=top_imps, y=top_features)
        plt.title('Top Descriptor Importances')
        
        plt.tight_layout()
        plt.savefig(f'{target}_{model_name}_plots.png')
        print(f"Plots saved as {target}_{model_name}_plots.png")
        plt.close()

if __name__ == "__main__":
    print("Initializing Automated Cheminformatics ML System...")
    
    # Dummy data for demonstration
    dummy_smiles = [
        "c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccccc1O", 
        "c1ccc(cc1)C=C", "C1=CC=CC=C1", "CC(=O)O", "CCO", "CCN",
        "c1cc(oc1)C=O", "c1ccsc1", "c1ccc2c(c1)ccc3ccccc23", "c1ccc(c(c1)C)C", "c1ccc(cc1)Cl",
        "c1cc(cc(c1)Cl)Cl", "C1CCCCC1", "c1ccc(cc1)N", "c1cc(ccc1C)N", "c1ccc(cc1)F"
    ]
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'SMILES': dummy_smiles,
        'HOMO': np.random.uniform(-6.5, -4.5, 20),
        'LUMO': np.random.uniform(-4.0, -2.0, 20),
        'Bandgap': np.random.uniform(1.0, 3.0, 20)
    })
    
    system = CheminformaticsMLSystem(target_properties=['HOMO', 'LUMO', 'Bandgap'])
    
    print("\nExtracting features for training dataset...")
    df_features = system.generate_features(dummy_data['SMILES'])
    df_train = pd.merge(dummy_data, df_features.drop('SMILES', axis=1), left_index=True, right_index=True)
    system.train_models(df_train)
    
    unseen_smiles = ["c1ccc(cc1)C#N", "c1cc(sc1)c2sccc2", "c1ccc2c(c1)oc3ccccc23", "INVALID_SMILES_TEST"]
    predictions = system.virtual_screening(unseen_smiles, rank_by='Bandgap', ascending=True)
    
    print("\n--- Ranked Candidates ---")
    print(predictions.to_string(index=False))
    
    print("\nGenerating Research-Grade Plots...")
    X_sample = df_train[system.all_feature_names]
    y_true = df_train['Bandgap']
    scaler = system.scalers['Bandgap']
    rf_model = system.models['Bandgap']['RF']
    y_pred = rf_model.predict(scaler.transform(X_sample))
    
    system.generate_visualizations('Bandgap', y_true, y_pred, 'RF', rf_model)
    print("\nFINAL GOAL ACHIEVED!")
