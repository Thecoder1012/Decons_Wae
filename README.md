# Deconstructing Wasserstein Autoencoders

<b> Step 1 : Install the requirements</b>
````
pip install -r requirements.txt
````

<b> Step 2 : Training the code</b>

<ins>Arguments</ins>

<b> **--groupsort**</b> Groupsort Activation function, default 0, if you want to initialise it type --groupsort 1

<b> **--js** </b> Jensen Shannon Divergence, default 0, if you want to initialise, type --js 1, if default then mmd is selected

<b> **--beta** </b> change latent space distribution to beta. Default 0, gaussian latent space is selected

<b> **--exp** </b> change latent space distribution to exponential. Default 0, gaussian latent space is seleted

<b> **--gauss** </b> Opt for gaussian ball

<b> **--mnist** </b> Opt for MNIST dataset.

To run the code please run
````
python3 train.py
````

All Robust codes have been given inside **Robust** folder for Gaussian Ball and Mnist. Each have been tested with **Cauchy**, **Dirichlet**, and **Gaussian** Noise.

**Experimentation** 

To experiment with Gaussian Ball position and increase the clusters we suggest to make changes inside <b>dataset.py</b>.

For experiment we have take a portion of 0.2 for mmd and js inclusion with reconstruction loss, user can change the portion in line <b> config.py </b>.

We offer open collaboration to the hyperparameters we have set. We also release all configuration settings in <b> config.py </b> for further experimentation.

The <b> model.py </b> presents a simple dense neural network model for mnist and gaussian ball reconstruction.

In robustness, we have mixed the datas with certain ratios mentioned in line **127, 129** in gaussian_ball **cauchy.py** and line **134, 137** in **dirichlet.py**. The ratio can be changed in line **95** and **101** respectively.
Similar experimentation can be done with MNIST. All configuration setting can be also set up to check the reconstruction.
