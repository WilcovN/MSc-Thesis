# MSc Thesis – Segmentation & Spatiotemporal Analysis

This repository contains a series of Jupyter Notebooks developed for the segmentation and analysis of biological imaging experiments conducted by **Wilco van Nes** and **Tsai-Ying Chen**. The entire project is implemented in Python, with detailed documentation embedded within each notebook.

Each notebook's name reflects its corresponding internal experiment and is independently documented to explain the logic, methodology, and computational environment used. All scripts are developed to run on **Intel CPU-based machines**.

> 🔍 The most developed code for the spatiotemporal analysis can be found in:
>  
> `Spatiotemporal code/TYC70_markov_phenotyping.ipynb`

---

## 📁 Repository Structure

```bash
.
├── Segmentation code/
│   ├── Strategy2_membran_segmentation.ipynb
│   ├── Strategy2_nucleus_segmentation.ipynb
│   ├── TYC70_WL_segmentation.ipynb
│   ├── TYC70_nucleus_segmentation_pipeline.ipynb
│   ├── TYC70_single_cell_segmentation_pipeline.ipynb
│   ├── WL_segmentation_TYC70_test.ipynb
│   └── test_cellpose.ipynb
├── Spatiotemporal code/
│   ├── Cell-cell_interaction_simulation.ipynb
│   ├── Markov_phenotyping_1ROI.ipynb
│   ├── Neighborhood_encoder_TYC70.ipynb
│   ├── Neighborhood_encoder_test.ipynb
│   └── TYC70_markov_phenotyping.ipynb
├── cellpose.yml
├── csbdeep.yml (used for stardist code)
└── README.md
