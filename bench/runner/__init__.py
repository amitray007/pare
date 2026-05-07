"""Subprocess-aware benchmark runner.

Captures wall, parent + children CPU (RUSAGE_CHILDREN), peak RSS for both,
and parallelism. CLI tools (mozjpeg, pngquant, oxipng, cjxl, gifsicle, ...)
do most of the work — measuring only the parent process underreports CPU
by an order of magnitude.
"""
