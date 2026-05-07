Automated Cheminformatics & Machine Learning Research Pipeline 🧪💻
An end-to-end, multi-phase automated workflow designed for molecular property prediction and organic photovoltaic (OPV) material discovery.
This repository is structured to demonstrate a complete progression from traditional machine learning to advanced AI architectures (Explainable AI and Graph Neural Networks).
🚀 Repository Structure & Phases
Phase 1: Core Cheminformatics & Machine Learning
* cheminformatics_ml_pipeline.py: The core engine. Handles SMILES validation, feature extraction (1D/2D descriptors, Morgan FPs), standard scaling, and dual-model training (Random Forest & XGBoost). Performs virtual screening and outputs metrics (R², RMSE, MAE).
* generate_structures.py: A dedicated modular script to automatically convert SMILES strings into 2D structural .png images.
Phase 2: Explainable AI (XAI)
* phase2_shap_explainability.py: Implements SHAP (SHapley Additive exPlanations) to interpret the XGBoost models. Breaks down the "black box" to explain exactly which molecular features drive the HOMO/LUMO/Bandgap predictions.
Phase 3: Advanced AI Architectures
* phase3_streamlit_webapp.py: A fully functional interactive Web Application. Allows users to input SMILES strings, visualize 2D structures, and get real-time property predictions in a user-friendly UI.
* phase3_gnn_pytorch.py: A Deep Learning template utilizing PyTorch Geometric. Defines a Graph Convolutional Network (GCN) architecture that treats molecules directly as graphs (Nodes = Atoms, Edges = Bonds) for next-generation material discovery.
🛠️ Installation
1. Clone this repository:
git clone https://github.com/zarbi8/Cheminformatics-ML-Pipeline.git
cd Cheminformatics-ML-Pipeline
2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required dependencies:
pip install -r requirements.txt
🖥️ Usage
1. Run the Core ML Pipeline:
python cheminformatics_ml_pipeline.py
2. Run the 2D Structure Generator:
python generate_structures.py
3. Generate Explainable AI (SHAP) Plots:
python phase2_shap_explainability.py
4. Launch the Interactive Web App:
streamlit run phase3_streamlit_webapp.py
📊 Outputs Expected
   * Console Metrics: Evaluative scores (R², RMSE) for trained models.
   * Images & Folders: Automatic creation of a Screened_Molecules_Structures/ folder with 2D drawings.
   * Visualizations: Actual vs. Predicted plots, error distributions, and SHAP Summary plots saved as .png files.
Built to accelerate high-throughput screening in materials science and computational chemistry.
