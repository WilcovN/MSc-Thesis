
# Chaotic_systems

Chaotic Systems is an AP3751 project focused on generating chaotic paths using a neural network. The chaotic trajectories analyzed here are derived from a Hénon-Heiles system. To achieve this, we employ a reservoir neural network. 

## Description
Chaotic Systems is a python repository composed of one python file filled with a neural network as class and a report jupyter notebook. The neural network that we made is called the `ReservoirNeuralNetwork` class and can be found within the `Hénon_Heiles.py`  and is used to create an untrained reservoir. `Hénon_Heiles.py` also contains all the functions required for analysis of trained reservoir networks, which we used in the report.

Available functions are described and used step-by-step in the Report jupyter notebook. Each section is designed to continue on previous results, thus we advice to run sections sequentially. All functions are explained in further detail in the `Hénon_Heiles.py` file. In the Report Jupyter notebook, the random seed is fixed to ensure consistent results for the final report version. In the final version, all cells are run with the fixed seed, However the energy conservation conlcusions were drawn after repeated runs without a fixed seed to ensure it generalizes across reservoirs.

## Getting Started

To run the Reservoir networks and analysis we perform on them one should first clone our repository:

    git clone https://gitlab.tudelft.nl/ap3751-projects/chaotic_systems.git

We use several standard libraries such as numpy, matplotlib, scipy, PIL (via Pillow), and IPython. The only package you might not have installed is reservoirpy, which can be installed using:

    pip install reservoirpy

We also have a `requirements.txt` file containing the dependencies of our repository. If you use anaconda, reservoirpy can only be installed using the pip package manager.
To create an anaconda environment that satisfies the dependencies run:

    conda create -n <environment-name> pip -y
    conda activate <environment-name>
    pip install -r requirements.txt
    python -m ipykernel install --user --name <environment-name> --display-name "environment-name"
    
from the anaconda prompt command window. This will make the anaconda environment available as a kernel within jupyter notebook.

If you use pip simply run:

    pip install -r requirements.txt


## Usage
As explained in the Description, please open the Report Jupyter notebook. Install the required packages to ensure every cell can run correctly. Then, read through the notebook and execute the code cells step-by-step.

If one does not care about evaluating the reservoir performance, a minimum viable implementation can be made using the instructions below:

Import the required libraries:

    import numpy as np
    import reservoirpy as rpy
    import matplotlib.pyplot as plt
    from Hénon_Heiles import *
    
    rpy.verbosity(0) 
    rpy.set_seed(42)  #make everything reproducible!


Create a reference trajectory from desired initial conditions using `HHsolution()` :

    t = np.linspace(0,1000,10000) #how many steps to calculate
    path = HHsolution(0.3,0,0.1,0.1,t) #(x,y,px,py,timesteps)

Process the reference trajectories:

    xP,yP, pxP, pyP = path
    
    #divide reference trajectories into training input
    xP_train = xP[:5000]
    yP_train = yP[:5000]
    pxP_train = pxP[:5000]
    pyP_train = pyP[:5000]

    #divide timeshifted reference trajectories into expected output
    xP_test = xP[1:5001]
    yP_test = yP[1:5001]
    pxP_test = pxP[1:5001]
    pyP_test = pyP[1:5001]

    #reshape it as reservoir expects (timesteps,features)
    training_input = np.concatenate((xP_train.reshape(-1,1),yP_train.reshape(-1, 1),pxP_train.reshape(-1,1),pyP_train.reshape(-1, 1)),axis=1)
    expected_output = np.concatenate((xP_test.reshape(-1,1),yP_test.reshape(-1, 1),pxP_test.reshape(-1,1),pyP_test.reshape(-1, 1)),axis=1)
    
  Create and train the reservoir using our `ReservoirNeuralNetwork()` class:

    rnn = ReservoirNeuralNetwork(n_neurons=2000, lr=0.34, sr=0.9, ridge=1e-7)
    rnn.train(training_input, expected_output,warmup=10)

To visualise the network trajectory you first have to make the predictions using our `predict_ntimesteps()` function. The input should be a short timeseries of the trajectory you want to predict consisting of the same features that you trained on:

    n_predictions = 10000 #how many steps you want to predict
    predicted_data = rnn.predict_ntimesteps(training_input[:50,:],n_predictions)
   After which you can use matplotlib or our built in functions to visualise the prediction:

    plt.plot(predicted_data[:,0], predicted_data[:,1], linewidth=0.3, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Reservoir Trajectory")

![](img/readme_reservoir_trajectory.png)



		


 
    

	


## Conclusion
The reservoir networks implemented in this repository are able to forecast a limited range of complex dynamics described by the Hénon Heiles system.  Stable dynamics can be forecasted forecasted accurately. Intermediate or chaotic dynamics, however, do not forecast correctly. In addition, our results do not conclusively demonstrate that reservoir networks can extract the most essential feature of Hamiltonian systems: Energy conservation.  While reservoir computing networks show promise, further research on network architectures is necessary to elucidate whether the full spectrum of Hamiltonian dynamics can be reproduced reliably.

## Project Contributions & Authors
Rocher 

 - Coded Primitive Reservoir Hamiltonian predictor
 - Reservoir architecture exploration
 - Coded Poincaré solver and plotter functions
 - Coded intitial condition solver functions
 - Coded Trajectory plotter for multiple energy levels
 - Coded Reservoir predictor in ReservoirNeuralNetwork class 
 - Wrote the reservoir predictions and poincaré sections of the report
 - Wrote Minimum viable implementation and some other parts of Readme
 - Adjusted Energy conservation section to incorporate Normalized energy deviation as in Zhang et al.
 - verified/wrote in report whether training the reservoir on energy improves results
 - Wrote Conclusion & Discussion
 


Jasper

 - Wrote introduction about Reservoir Neural networks
 - Wrote code to generate predicted data and analyse the mean square error per timepoint
 - Wrote code and text about the analysis of hyperparameters
 - Implemented a 2D and 4D meshgrid analysis to find the most suitable hyperparameters that
   minimize the MSE for the given initial conditions. 

Wilco

 - Wrote and coded the Introduction, Hénon-Heiles, Energy conservation sections and functions
 - Set up and maintain the skeleton of the Report and Gitlab repository
 - Create the ReservoirNeuralNetwork class 
 - peer reviewed all parts

Zan

 - Implemented the analysis of robustness.
 - Coded some functions in `Hénon_Heiles.py` and `robustness.py` to generate data with noise. Used the noisy data to train the model or as input for predicting the dynamics of chaotic system to evaluate the impact of data noise on the model's prediction perfformance.
 - Trained the reservior network using a mixture of data from different energy levels and various trajectories to enhance the model's ability to accurately predict inputs across a range of conditions, thereby improving its robustness and overall predictive performance.

Elias 
 - First General Hénon Heiles Hamiltonian reservoir training loop

-----------



