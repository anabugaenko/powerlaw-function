# powerlaw-function: A Python Package for Analysing Heavy-Tailed Data

This Package provides functionality for fitting power law functions to random variables. 

In [Clauset, Shalizi and Newman](https://arxiv.org/abs/0706.1062) and [Virkar and Clauset. 2012](https://arxiv.org/abs/1208.3524), they provide methods appropriate for testing whether data are drawn iid from a power-law distribution. To do this, they use kernel density estimation (KDE) to infer a CDF from a finite sample of data, and then fit a power-law to the inferred CDF through the process of distribution fitting, where they model the probability distribution of a single variable (see - [mathworks](https://www.mathworks.com/help/stats/curve-fitting-and-distribution-fitting.html)).

In our case, we are interested in modelling a response variable as a function of a predictor variable and providing methods for testing whether such a stochastic process follows a power law. For instance, we can infer an autocorrelation function (ACF) from a finite time-series sample, and then we can fit a power-law to the inferred ACF to infer whether the process has long-memory. Our package provides core functionality to perform the fitting and assess the goodness of fit in such cases. It does not provide the functionality for computing confidence intervals, as the correct way to do this will depend on the specific use case.


# Installation 

We recommend Conda for managing Python packages; pip for everything else. To get started, install `pip` and run:

  `pip install powerlaw_function`

# Further Development

`powerlaw-function is open for further development by the community. If there are additional features you'd like to see developed, feel free to raise an issue and/or pull request.

# Acknowledgement 

We want to extend a thank you to Jeff Alstott, Ed Bullmore, Dietmar Plenz for open-sourcing the 'powerlaw' Package which can be used to model the PDF of a single variable. Their implementations provided a crucial starting point for the development `powerlaw-function`. We would also like to Andreas Klaus, Aaron Clauset, Cosma Shalizi, and Adam Ginsburg for their original paper [POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA](https://arxiv.org/abs/0706.1062)  which serves as a theoretical basis for both approaches.

# How to Cite

  Academics, please cite as:
  
    @article{abuga2023power,
      title={Power-laws functions in Empirical Data}
      author={Ana Bugaenko and Christian Wayi-Wayi},
      journal={In preparation},
      year={TBA}
    }


