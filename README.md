# Protein Binding Site Prediction with DDPM-Augmented Graph Neural Networks

This project aims to accurately predict residue-level protein binding sites using a Graph Neural Network (GNN) pipeline enhanced by a DDPM-based generative model. It integrates large-scale protein embeddings (ESM-2), deep edge prediction, and class-imbalanced data augmentation for robust residue classification.
## 🔍 Overview
### Key Components:
- **ESM-2 Representation**: Residue-level embeddings using Meta’s ESM2 (650M).
- **DDPM Diffusion Generator**:Generates high-quality synthetic positive (binding site) residues using a denoising diffusion probabilistic model (DDPM), followed by similarity filtering.
- **Similarity Filtering**: Only generated residues highly similar to real positive samples are retained (via cosine similarity thresholding).
- **Edge Predictor**: Connects generated and real nodes using an MLP-based predictor to form an enhanced graph G*.
- **GCN + MLP Classifier**:Classifies residues based on learned neighborhood structure and concatenated ESM features.
- **Focal Loss**:Effectively handles severe class imbalance between binding and non-binding residues.
- ## 📁 Project Structure
- 
├── main.py                     # Entry script
├── Raw_data/                  # Input sequences and label files
├── modules/
│   ├── data_loader.py         # Loads and embeds protein sequence data
│   ├── ddpm_diffusion_model.py# DDPM-based positive sample generator (NEW)
│   ├── edge_predictor.py      # Builds edges between nodes (real + generated)
│   └── model.py               # GCN-based classifier, training & evaluation
├── Weights/
│   └── best_model.pt          # Trained model checkpoint
└── README.md


## ⚙️ Dependencies

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)  
- `fair-esm (for ESM-2) 
- scikit-learn  
- matplotlib, tqdm (optional)

Install requirements:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install fair-esm scikit-learn tqdm
🚀 How to Run
1.Prepare raw data in Raw_data/ folder:
>protein_name
SEQUENCE...
LABELS...(0 or 1 per residue)
2.Run the pipeline:
python main.py
3.Output includes:
Training logs with loss & accuracy

Final test results (Accuracy, F1, MCC)

best_model.pt as the saved model
📊 Example Result
Train Accuracy: 96.4%
Test Accuracy : 87.5%
F1 Score      : 0.39
MCC           : 0.37

📌 Highlights
✅ DDPM-powered augmentation: More principled and controllable generation of residue-level binding site candidates.
✅Cosine similarity screening: Ensures generated residues resemble real ones to improve generalization.
✅ ESM + GNN fusion: Leverages transformer embeddings and structural cues together.
✅ Edge learning with top-k filtering: Adds meaningful connectivity to synthetic data.
✅ Focal loss + class balancing: Handles real-world skewed datasets effectively.
✏️ Citation
f you use this code or idea in your work, please cite or acknowledge:

"DDPM-Augmented Graph Learning for Protein Binding Site Prediction"

📬 Contact
For questions or collaborations, contact:
hanqing zhang
Email: 3165619783@qq.com

