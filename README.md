# Surrogate Models for Storm Surge Prediction

## **Overview**
This project focuses on developing and exploring surrogate models for storm surge predictions using data-driven approaches. The goal is to predict storm surge time-series over 3,000 locations using efficient dimensionality reduction techniques and neural network-based models, particularly AutoEncoders. This work is aimed at addressing the computational challenges posed by high-fidelity numerical simulations.

---

## **Objectives**
1. Develop data-driven surrogate models to emulate high-fidelity storm surge simulations.
2. Use dimensionality reduction techniques, such as AutoEncoders, to capture nonlinear relationships in the dataset.
3. Apply Gaussian Process metamodels to enhance prediction accuracy.
4. Investigate and optimize neural network architectures using advanced hyperparameter optimization techniques.

---

## **Dataset**
The dataset is derived from the North Atlantic Coast Comprehensive Study (NACCS) conducted by the U.S. Army Engineer Research and Development Center, Coastal and Hydraulics Laboratory (ERDC-CHL). It includes:
- **595 landfalling storms**, grouped into **4 track groups** corresponding to 89 Master Tracks (MTs).
- Parameters characterizing each storm:
  - **Landfall Location**: Latitude (`xlat`), Longitude (`xlon`).
  - **Heading Direction**: (`β`).
  - **Central Pressure Deficit**: (`ΔP`).
  - **Translational Speed**: (`vts`).
  - **Radius of Maximum Wind Speed**: (`Rmw`).
- Simulation data for **18,977 save points (SPs)**.

---

## **Project Structure**
The project is organized into the following files and folders:

### **Code**
1. **Centroid_Process**: 
   - Code for centroid calculation and validation.
2. **Model_Exploration**: 
   - `Model_Summary.py`: Summarizes model architectures.
   - `Latent_reconstruction_Extraction.py`: Extracts latent and reconstructed data from the decoder.
   - `Latent_to_Prediction_and_Reconstruction.py`: Generates predictions from latent data using the decoder.
3. **Optuna_Hyperparametrization**: 
   - Python and batch files for hyperparameter optimization of activation functions (tanh, sigmoid, selu, softplus) using Optuna.
4. **Plots**: 
   - Scripts to create all plots for the thesis, including:
     - Activation functions.
     - PCA and AE reconstruction.
     - GP predictions.
     - Database visualizations.
     - Spatial distribution for specific regions.
5. **Trials_Model**: 
   - Python and batch files for trials selected from Optuna results.
6. **problematiclocationcalculation.py**: 
   - Code to identify and visualize problematic nodes.



### **Optuna_Trial_Results**
- Results of Optuna hyperparameter optimization for various activation functions.



---

## **Key Files**
1. **Architecture_and_Validationmetric_Results.CSV**: 
   - Contains the architecture of all explored models, their hyperparameters, and validation metrics.
   - Highlights the model used for further validation.
2. **ClusterNum_latlong_nodeindices.CSV**: 
   - Includes the 3,000 cluster representatives with their indices, latitude/longitude values, and distances from the cluster centroids.

---

## **Deliverables**
1. Dimensionality reduction using AutoEncoders to handle nonlinear patterns in the dataset.
2. Streamlined surrogate models for spatiotemporal storm surge prediction.
3. Validation of models using reconstructed GP predictions.
4. Comprehensive analysis and visualization of results, including spatial distributions and problematic node identification.

