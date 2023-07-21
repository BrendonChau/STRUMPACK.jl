module STRUMPACK

using Libdl, MPI, LinearAlgebra
import Libdl: dlsym, dlopen

export CSPOptions,
       SP_d_struct_default_options,
       SP_d_struct_from_dense,
       SP_d_struct_from_elements,
       SP_d_struct_mult,
       SP_d_struct_solve,
       SP_d_struct_factor,
       SP_d_struct_shift,
       SP_d_struct_rank,
       SP_d_struct_destroy,
       SP_d_struct_memory,
       SP_d_struct_nonzeros,
       SP_create_kernel_double,
       SP_kernel_fit_HSS_double,
       SP_destroy_kernel_double

const libstrumpack = "/Users/brendonchau/bin/STRUMPACK/install/lib/libstrumpack.dylib"

include("dense.jl")
include("kernel.jl")
# include("sparse/sparse.jl")
# include("sparse/sparseopts.jl")

end