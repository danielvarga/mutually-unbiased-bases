cat triplets.e-2x | while read f ; do echo -n "$f " ; python hadamard_cube.py $f ; echo ; done > is_fourier.cout 2> is_fourier.cerr
