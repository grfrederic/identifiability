import diffrax


# DEFAULT_SOLVER = diffrax.Kvaerno5()
DEFAULT_SOLVER = diffrax.Dopri5()

DEFAULT_STEPSIZE_CONTROLLER_KWARGS = dict(
    dtmax=0.1,
    factormin=0.1,
    rtol=1e-5,
    atol=1e-5,
)
DEFAULT_STEPSIZE_CONTROLLER = diffrax.PIDController(
    **DEFAULT_STEPSIZE_CONTROLLER_KWARGS
)
