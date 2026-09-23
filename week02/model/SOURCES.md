# Open-source design references

This directory does not vendor or copy third-party source. The local model is an
independent, compact implementation that adapts the following public ideas:

- [RVO2 / ORCA](https://github.com/snape/RVO2), Apache-2.0: represent local
  collision avoidance as selection of a velocity close to a preferred velocity,
  under constant-velocity predictions of nearby agents. ORCA itself solves
  half-plane constraints; this project instead evaluates a finite candidate set.
- [JuPedSim](https://github.com/PedestrianDynamics/jupedsim), LGPL-3.0: expose
  perception, future-position prediction and strategy/action selection as
  separate parts of the Anticipation Velocity Model. This project uses the same
  conceptual separation but not JuPedSim source code.
- [OpenTraj](https://github.com/crowdbotp/OpenTraj), MIT: repository structure and
  trajectory-data context.

The experiment-specific right-side route prior comes from the accompanying
circle-antipode data analysis, not from those libraries.
