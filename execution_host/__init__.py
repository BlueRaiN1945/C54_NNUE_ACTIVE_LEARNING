"""Scripts designed to run ON the execution host (BigPC), never on Medium-PC.

This package is deliberately outside medium_pc_audit: that package's own
charter states it "does not run chess engines, does not train NNUE networks,
and does not install/build Stockfish, c-chess-cli, or Docker" (README.md).
Anything that spawns an engine subprocess belongs here instead, kept visibly
separate so the boundary is structural, not just a convention to remember.

Nothing in this package is executed by this repository's tests or by any
Medium-PC control-plane code. Only pure, engine-free functions (parsing,
validation) are imported and unit-tested from tests/. Deploying a script from
this package to BigPC and running it there requires separate, explicit
authorization -- writing or testing the file here does not constitute that.
"""
