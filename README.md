# powerlaw-function: A Python Package for Analysing Heavy-Tailed Data

This repo provides functionality for fitting power law functions to random variables. Some example use-cases are:

1.  Testing whether data are drawn iid from a power-law distribution.  To do this, we can use, e.g. KDE, to infer the CDF from a finite sample of data, and then we can fit a power-law to the inferred CDF, similarly to [Clauset et al. 2007](https://arxiv.org/abs/0706.1062) and [Virkar and Clauset. 2012](https://arxiv.org/abs/1208.3524).
2.  Testing whether a stochastic process follows a power law. For instance, we can infer an autocorrelation function from a finite time-series sample, and then we can fit a power-law to the inferred ACF to infer whether the process has long memory. 

`powerlaw-function` provides the core functionality to perform the fitting part of the process in these kinds of use-cases.  It does not provide the functionality for computing confidence intervals, as the correct way to do this will depend on the specific use-case.

# Installation 

We recommend Conda for managing Python packages; pip for everything else. To get started, simply install `pip` and run:

  `pip install powerlaw_function`

# Further Development

`powerlaw-function is open for further development by the community.  Feel free to raise an issue if you find a problem. If there are additional features you'd like to see developed, please [raise an issue](https://github.com/anabugaenko/powerlaw-function/issues) and/or pull request; this repository is actively being developed and any tickets will be addressed in order of importance. 

# Acknowledgement 

We would like to extend a thank you to Jeff Alstott, Ed Bullmore, Dietmar Plenz for open-sourcing the 'powerlaw' Package, their implementations provided a crucial starting point for the development `powerlaw-function`. We would also like to Andreas Klaus, Aaron Clauset, Cosma Shalizi, and Adam Ginsburg for their original paper [POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA](https://arxiv.org/abs/0706.1062)  which serves as a theoretical basis for our approach.

# How to Cite

  Academics, please cite as:
  
    @article{abuga2023power,
      title={Power-laws functions in Empirical Data}
      author={Ana Bugaenko and Christian Wayi-Wayi},
      journal={In preparation},
      year={TBA}
    }


