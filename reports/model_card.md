# Model card

The selected system is a shared compound-dose shift of the four module scores. It is not a
flow. On validation the flow and the conditional Gaussian both lost to this shift, so they
were not carried to the test ranking.

A cell's membership in a goal is a logistic curve centered at a training-solvent percentile,
as wide as the spread of those solvent scores. Utility is the drug's average membership
minus the solvent call's, minus the same difference for apoptosis. Doing nothing has utility
zero and wins ties.

The test ranking was hashed before test-treated scores were computed. On 207 test queries
the shift's mean top-1 utility is 0.022, which is 0.08 points above ridge and 2.2 points
above doing nothing. The prespecified +5 point gate does not pass. Plate 14, held out of
fitting, shows a smaller gain: 0.034 when the condition was seen on plate 6, and 0.013 when
the pair was withheld everywhere.

The page in `app/index.html` shows one line on one plate. It refuses a comparison across
doses. Measured utilities stay hidden until asked for, and they come from the locked test
file rather than a new fit.
