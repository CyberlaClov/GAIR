from reliability.Fitters import Fit_Weibull_2P

# Given data
failures = [229, 386, 180, 168, 122, 138]
right_censored = [309, 104, 217, 167]  # Assuming no right censored data provided

# Fit Weibull distribution
fit = Fit_Weibull_2P(
    failures=failures, right_censored=right_censored, show_probability_plot=False
)

# Extract parameters
beta = fit.distribution.beta
eta = fit.distribution.alpha
# Assuming t0 = 0 as per common practice unless specified otherwise
print(beta, eta, 0)
