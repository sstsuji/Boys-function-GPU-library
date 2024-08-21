# Boys-function-GPU-library
A GPU library for high-throughput evaluation of the Boys function


## The Boys function
$$F_n(x) = \int_{0}^{1} t^{2n}e^{-xt^2} dt$$
- $n$: non-negative integer
- $x$: non-negative real number


## Requirements
- Hardware
    - OS: Linux
    - CPU: x86-64 architecture
    - GPU: CUDA-enabled GPUs (NVIDIA Volta architecture or later)
- Software
    - GCC 9.4.0 or later
    - CUDA Toolkit 11.2 or later
    - GMP 6.2.1 or later (The GNU Multiple Precision Arithmetic Library)


## Limitations
- Input range for the Boys function: $0 \le n \le 24$
- Number of input pairs $(n, x)$ is constrained by the available GPU global meomry size


## Installation

### GMP
- Build the multiple precision arithmetic library for numerical tests of the Boys function
```bash
# Download the GMP source from: https://gmplib.org/#DOWNLOAD
# Reference for this installation: https://gmplib.org/manual/Installing-GMP
./configure --prefix=/path/to/gmp/root
make
make check
make install

export GMP_ROOT=/path/to/gmp/root
```

### Boys function GPU library
- Source files in `src/` are individually compiled to object files in `obj/`
- Linked executable binary files are generated in `bin/`
```bash
git clone https://github.com/sstsuji/Boys-function-GPU-library.git
cd Boys-function-GPU-library
make
```


## Usage


## Experimental results


## License
This library is dual-licensed, under the conditions of the GNU Lesser General Public License version 3 (LGPL-3.0), and the GNU General Public License version 2 (GPL-2.0).
