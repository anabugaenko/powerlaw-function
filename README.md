# powerlaw-function: A Python Package for Analysing Heavy-Tailed Functions

`powerlaw-function` is a flexible Package that extends the statistical methods developed by [Clauset et al. 2007](https://arxiv.org/abs/0706.1062) and [Virkar and Clauset. 2012](https://arxiv.org/abs/1208.3524) for distribution fitting; i.e., fitting and determining whether a probability distribution of a single variable follows a power law.</b>
The `powerlaw` package is available at [powerlaw](https://github.com/jeffalstott/powerlaw/tree/master).

However,  here we are instead interested in modelling a response variable as a function of a predictor variable by finding the best functional approximation. In this context, we are fitting power law functions to known functions, which is closer to the realm of approximation theory and deterministic modelling as opposed to probabilistic (or stochastic) modelling approaches. 


# Installation 

We recommend conda for managing Python packages; pip for everything else. To get started, simply install `pip` and run:

  `pip install powerlaw-function`

# Basic Usage 

This tells you everything you need to know for the simplest, typical use cases:

    import powerlaw-function

    # Data
    xy_df = pd.DataFrame({
        'x_values': x_values,
        'y_values': y_values
    })
    
    results = Fit(xy_df, verbose=True)
    R, p = results.function_compare('pure_powerlaw', 'exponential_function')
    
    print('Alpha:', results.pure_powerlaw.params.alpha)
    print('xmin:', results.pure_powerlaw.xmin)
    print('BIC:', results.pure_powerlaw.bic)
    print('Adjusted R-squared:', results.pure_powerlaw.adjusted_rsquared)
    print(f'Likelihood Ratio: {R}, p.value: {p}')

For more details on how to use the Package, including figures and approach, see the Manuscript and Article Bugaenko et al. 2023 respectively, which illustrate all of the `powerlaw-function` features and provide the mathematical background underlying these methods.

# Power Laws vs. Alternative Functions

TODO

# Further Development

`powerlaw-function is open for further development by the community.  Feel free to raise an issue if you find a problem. If there are additional features you'd like to see developed, please [raise an issue](https://github.com/anabugaenko/powerlaw-function/issues) and/or pull request; this repository is actively being developed and any tickets will be addressed in order of importance. 

# Acknowledgement 

We would like to extend a thank you to Jeff Alstott, Ed Bullmore, Dietmar Plenz for open sourcing the 'powerlaw' Package, their implementations provided a crucial starting point for making `powerlaw-function`. We would also like to Andreas Klaus, Aaron Clauset, Cosma Shalizi, and Adam Ginsburg for their original paper [POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA](https://arxiv.org/abs/0706.1062)  which serves as a theoretical basis for our approach.

# How to Cite

  Academics, please cite as:
  
    @article{abuga2023power,
      title={Power-law functions in deterministic data,
      author={Ana Bugaenko and Christian Wayi-Wayi},
      journal={In preparation},
      year={2023}
    }


