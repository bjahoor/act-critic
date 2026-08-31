"""ACT with a critic head: the policy's action chunk and a failure score from one forward pass.

The trunk is LeRobot's ACT, frozen. The head reads the encoder output and never
touches the decoder, so the policy's behaviour is unchanged.
"""
