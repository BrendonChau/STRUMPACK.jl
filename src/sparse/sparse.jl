@enum STRUMPACK_PRECISION::Cint begin
    STRUMPACK_FLOAT=0
    STRUMPACK_DOUBLE=1
    STRUMPACK_FLOATCOMPLEX=2
    STRUMPACK_DOUBLECOMPLEX=4
    STRUMPACK_FLOAT_64=5
    STRUMPACK_DOUBLE_64=6
    STRUMPACK_FLOATCOMPLEX_64=7
    STRUMPACK_DOUBLECOMPLEX_64=8
end

@enum STRUMPACK_INTERFACE::Cint begin
    STRUMPACK_MT=0
    STRUMPACK_MPIDIST=1
end

@enum STRUMPACK_COMPRESSION_TYPE::Cint begin
    STRUMPACK_NONE=0
    STRUMPACK_HSS=1
    STRUMPACK_BLR=2
    STRUMPACK_HODLR=3
    STRUMPACK_BLR_HODLR=4
    STRUMPACK_ZFP_BLR_HODLR=5
    STRUMPACK_LOSSLESS=6
    STRUMPACK_LOSSY=7
end

@enum STRUMPACK_MATCHING_JOB::Cint begin
    STRUMPACK_MATCHING_NONE=0
    STRUMPACK_MATCHING_MAX_CARDINALITY=1
    STRUMPACK_MATCHING_MAX_SMALLEST_DIAGONAL=2
    STRUMPACK_MATCHING_MAX_SMALLEST_DIAGONAL_2=3
    STRUMPACK_MATCHING_MAX_DIAGONAL_SUM=4
    STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING=5
    STRUMPACK_MATCHING_COMBBLAS=6
end

@enum STRUMPACK_REORDERING_STRATEGY::Cint begin
    STRUMPACK_NATURAL=0
    STRUMPACK_METIS=1
    STRUMPACK_PARMETIS=2
    STRUMPACK_SCOTCH=3
    STRUMPACK_PTSCOTCH=4
    STRUMPACK_RCM=5
    STRUMPACK_GEOMETRIC=6
    STRUMPACK_AMD=7
    STRUMPACK_MMD=8
    STRUMPACK_AND=9
    STRUMPACK_MLF=10
    STRUMPACK_SPECTRAL=11
end

@enum STRUMPACK_GRAM_SCHMIDT_TYPE::Cint begin
    STRUMPACK_CLASSICAL=0
    STRUMPACK_MODIFIED=1
end

@enum STRUMPACK_RANDOM_DISTRIBUTION::Cint begin
    STRUMPACK_NORMAL=0
    STRUMPACK_UNIFORM=1
end

@enum STRUMPACK_RANDOM_ENGINE::Cint begin
    STRUMPACK_LINEAR=0
    STRUMPACK_MERSENNE=1
end

@enum STRUMPACK_KRYLOV_SOLVER::Cint begin
    STRUMPACK_AUTO=0
    STRUMPACK_DIRECT=1
    STRUMPACK_REFINE=2
    STRUMPACK_PREC_GMRES=3
    STRUMPACK_GMRES=4
    STRUMPACK_PREC_BICGSTAB=5
    STRUMPACK_BICGSTAB=6
end

@enum STRUMPACK_RETURN_CODE::Cint begin
    STRUMPACK_SUCCESS=0
    STRUMPACK_MATRIX_NOT_SET=1
    STRUMPACK_REORDERING_ERROR=2
    STRUMPACK_ZERO_PIVOT=3
    STRUMPACK_NO_CONVERGENCE=4
    STRUMPACK_INACCURATE_INERTIA=5
end

mutable struct STRUMPACK_Sparse_Solver
    solver    :: Any
    precision :: STRUMPACK_PRECISION
    interface :: STRUMPACK_INTERFACE
    STRUMPACK_Sparse_Solver() = new()
end

function STRUMPACK_init_mt(
    S::STRUMPACK_Sparse_Solver,
    precision::STRUMPACK_PRECISION,
    interface::STRUMPACK_INTERFACE,
    argv::Vector{String},
    verbose::Int
    )
    argc = length(argv)
    ccall((:STRUMPACK_init_mt, libstrumpack), 
          Cvoid, 
          (Ref{STRUMPACK_Sparse_Solver}, 
           Cint, 
           Cint, 
           Cint, 
           Ptr{Ptr{UInt8}},
           Cint), 
           S, precision, interface, argc, argv, verbose)
end

# function STRUMPACK_init_mt(
#     S::Ref{Ptr{Cvoid}},
#     precision::STRUMPACK_PRECISION,
#     interface::STRUMPACK_INTERFACE,
#     argv::Vector{String},
#     verbose::Int
#     )
#     argc = length(argv)
#     ccall((:STRUMPACK_init_mt, libstrumpack), 
#           Cvoid, 
#           (Ref{Ptr{Cvoid}}, 
#            Cint, 
#            Cint, 
#            Cint, 
#            Ptr{Ptr{UInt8}},
#            Cint), 
#           S, precision, interface, argc, argv, verbose)
# end

function STRUMPACK_destroy(S::Ref{Ptr{Cvoid}})
    ccall((:STRUMPACK_destroy, libstrumpack), 
          Cvoid, 
          (Ref{Ptr{Cvoid}},), 
          S)
end

function STRUMPACK_destroy(S::STRUMPACK_Sparse_Solver)
    ccall((:STRUMPACK_destroy, libstrumpack), 
          Cvoid, 
          (Ref{STRUMPACK_Sparse_Solver},), 
          S)
end

function STRUMPACK_set_csr_matrix(
    S::Ref{Ptr{Cvoid}},
    N::Int,
    row_ptr::AbstractVector{Int32},
    col_ind::AbstractVector{Int32},
    values::AbstractVector{Float64},
    symmetric_pattern::Int
    )
    ccall((:STRUMPACK_set_csr_matrix, libstrumpack),
          Cvoid,
          (Ptr{Cvoid}, Cint, Ref{Cint}, Ref{Cint}, Ref{Cdouble}, Cint),
          S[], N, row_ptr, col_ind, values, symmetric_pattern)
end

function STRUMPACK_set_csr_matrix(
    S::STRUMPACK_Sparse_Solver,
    N::Int,
    row_ptr::AbstractVector{Int32},
    col_ind::AbstractVector{Int32},
    values::AbstractVector{Float64},
    symmetric_pattern::Int
    )
    ccall((:STRUMPACK_set_csr_matrix, libstrumpack),
          Cvoid,
          (Ref{STRUMPACK_Sparse_Solver}, Cint, Ref{Cint}, Ref{Cint}, Ref{Cdouble}, Cint),
          S, N, row_ptr, col_ind, values, symmetric_pattern)
end

function STRUMPACK_reorder_regular(
    S::Ref{Ptr{Cvoid}},
    nx::Int,
    ny::Int,
    nz::Int,
    components::Int,
    width::Int
)
    ccall((:STRUMPACK_reorder_regular, libstrumpack),
          Cint,
          (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint),
          S[], nx, ny, nz, components, width)
end

function STRUMPACK_factor(
    S::Ref{Ptr{Cvoid}}
    )
    ccall((:STRUMPACK_factor, libstrumpack),
          Cint,
          (Ptr{Cvoid},),
          S[])
end

function STRUMPACK_solve(
    S::Ref{Ptr{Cvoid}},
    b::AbstractVector{Float64},
    x::AbstractVector{Float64},
    use_initial_guess::Int
    )
    ccall((:STRUMPACK_solve, libstrumpack),
          Cint,
          (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Cint),
          S[], b, x, use_initial_guess)
end