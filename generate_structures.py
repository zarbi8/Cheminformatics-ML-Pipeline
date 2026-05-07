import os
import warnings
from rdkit import Chem
from rdkit.Chem import Draw

warnings.filterwarnings('ignore')

def validate_and_draw_smiles(smiles_list, folder_name="molecule_structures"):
    """
    Validates SMILES strings and generates 2D molecular structures,
    saving them as PNG files in the specified folder.
    """
    print(f"\n--- Generating 2D Structures in '{folder_name}' folder ---")
    os.makedirs(folder_name, exist_ok=True) 
    valid_count = 0
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            filename = os.path.join(folder_name, f"molecule_{i+1}.png")
            Draw.MolToFile(mol, filename, size=(300, 300))
            valid_count += 1
        else:
            print(f"Warning: Invalid SMILES skipped -> {smi}")
            
    print(f"Successfully saved {valid_count} molecule images in '{folder_name}'!")

if __name__ == "__main__":
    print("Initializing 2D Structure Generator...")
    sample_smiles = [
        "c1ccc(cc1)C#N", 
        "c1cc(sc1)c2sccc2", 
        "c1ccc2c(c1)oc3ccccc23", 
        "INVALID_SMILES_TEST" 
    ]
    validate_and_draw_smiles(sample_smiles, folder_name="Screened_Molecules_Structures")
    print("Process complete.")
