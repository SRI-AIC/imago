# Reaver Assault CAML Y2 

This details how to train, perturb over the trajectories played by the CAML Y2 agent on the Reaver Assault task.

## Preliminaries

This presumes the directory structure where all caml projects are under caml/

    caml/imago
    CAML_ROOT = directory caml/ project is set.
    IMAGO_ROOT = $CAML_ROOT/imago

## Data and Model Files Setup
We now detail setup for the data and model files used in this package.
## Trajectory Data
The default directory for the interestingess data points is at,

    $CAML_ROOT/datasets/ReaverAssaultYear2

and should contain the files

    interaction_data.pkl.gz
    interestingness.csv.gz

NOTE: When the data is first loaded, the code will place several *.npz
caches in the same directory as these files, in order to speed up
further experiments.  These are large, potentially up to 20G. 

We will look to make this more efficient in future releases.

### Model Files
The model file to use should be placed at the following filepath, 

   $IMAGO_ROOT/models/ReaverAssaultYear2/model.pt

## Training

Use the invocation

    cd $IMAGO_ROOT
    python -m pvae.domains.reaverassaulty2.train

This will emit the model outputs to `$CAML_ROOT/output/ReaverAssaultYear2_*`

where the suffix is a combination of the selected learning rate and other parameters.

## Counterfactuals
Counterfactual identification can be performed via several methods,
- Multivariate Optimization
- Neighbors Analysis

### Python
The main class used to generate the Counterfactual experiments is 

        cf_exp = CFExps(sc2_domain, perturb, train_ds)

- CFExps is an instance of the imago.perturb.cfs.CFExps class, which accepts a domain specifier, desired perturbation, and training set to draw upon for NUNs.  This is the main class used to perform the queries and 
- perturb is an instance of the imago.perturb.Perturb class, which specifies the outcome variable and target variable.



### Configuration
All of the counterfactual perturbation methods share common configuration files.  
These are under imago/domains/reaverassaulty2/configs/*.yaml.  The following describes
configuration settings common to all counterfactual/perturbation methods.

To specify which episodes and frames to use as queries, add these as entries under the
'scenes' field, where the episode name is the field, and frame offset is the
value.  For example,

    scenes:
       ep0: 1
       ep10: 5

This will perform counterfactual selection using the following query points, represented
by the episode name and frame number ('ep0' frame 1, 'ep10' frame 5).  The variable to
identify a counterfactual on, and the desired direction of change, are specified by the 
'varname' and 'direction' fields.

'varname' is the column name of the interestingness variable to use, and 
direction is -1/+1 and represents the direction to perturb that variable.

    varname: Riskiness
    direction: 1

In this example, we will attempt to maximize Riskiness.

Results are specified in the results_dir field.

model_fpath, data_fpath, and intr_csv_fpath point to the VAE model file,
the interaction data, and interestingness CSV file to use.

### Multivariate Optimization

This implements multivariate optimization to identify a counterfactual for the given query point.  The optimizer will attempt to identify a nearby counterfactual in latent space that perturbs just the selected variable in the desired direction.   

To run this, 

    cd $IMAGO_ROOT
    python -m imago.domains.reaverassaulty2.multivar_opt_cf imago/domains/reaverassaulty2/config/config.yaml

Where the first argument to the class is the YAML file that specifies
the episode and frame offsets to use, as well as other configuration settings.

#### Configuration
The type of multivariate optimization scheme is selected by the 'method' key.  
Current valid values are,

- nm, Nelder-Mead simplex search with a Gaussian ball around query point as an initial simplex
- nm_strict, Nelder-Mead simplex search
- cma, CMA-ES particle search

### Neighborhood 
This approach uses a nearest neighbors, in feature-space, to train
a direction of perturbation in latent space corresponding to the desired change in the selcted variable.

To run this,

    cd $IMAGO_ROOT
    python -m imago.domains.reaverassaulty2.neighbor_cf_analysis imago/domains/reaverassaulty2/config/config.yaml

#### Configuration
The number of nearest neighbors is configured by the 'neighbors' field.  

### XAI World Experiments
To recreate the XAI World experiments, please see,

Likelihood results:
notebooks/220602_likelihood_verification/220602_likelihood_verification.ipynb

Simulator results:
notebooks/220502_highlights/220507_cartpole_cfs.py
notebooks/220502_highlights/220508_canniballs_cfs.py
notebooks/220502_highlights/220523_ray2_cfs_realism.py
notebooks/220502_highlights/220601_ray2_split_model_cfs.py
notebooks/230417_xai/230418_xaitest.ipynb