module STRUMPACK

using Libdl, MPI, LinearAlgebra
import Libdl: dlsym, dlopen

export CSPOptions,
       SP_d_struct_default_options,
       SP_d_struct_from_dense,
       SP_d_struct_from_elements,
       SP_d_struct_mult,
       SP_d_struct_factor,
       SP_d_struct_rank,
       SP_d_struct_destroy

const libstrumpack = "/Users/brendonchau/bin/STRUMPACK/install/lib/libstrumpack.dylib"

@enum SP_STRUCTURED_TYPE::Cint begin
    SP_TYPE_HSS=0
    SP_TYPE_BLR=1
    SP_TYPE_HODLR=2
    SP_TYPE_HODBF=3
    SP_TYPE_BUTTERFLY=4
    SP_TYPE_LR=5
    SP_TYPE_LOSSY=6
    SP_TYPE_LOSSLESS=7
end

mutable struct CSPOptions
    type      :: SP_STRUCTURED_TYPE
    rel_tol   :: Cdouble
    abs_tol   :: Cdouble
    leaf_size :: Cint
    max_rank  :: Cint
    verbose   :: Cint
    CSPOptions() = new()
end

struct CSPStructMat
    ref::Ref{Ptr{Cvoid}}
    CSPStructMat() = new(Ref{Ptr{Cvoid}}())
end

function SP_d_struct_default_options(opts::CSPOptions)
    ccall((:SP_d_struct_default_options, libstrumpack), Cvoid, (Ref{CSPOptions},), opts)
end

function SP_d_struct_from_dense(
    obj::Ref{Ptr{Cvoid}},
    A::AbstractMatrix{Float64}, 
    opts::CSPOptions)
    m, n = size(A)
    success = ccall((:SP_d_struct_from_dense, libstrumpack), 
                    Cint, 
                    (Ref{Ptr{Cvoid}}, Cint, Cint, Ref{Cdouble}, Cint, Ref{CSPOptions}), 
                    obj, m, n, A, m, opts)
    success == 0 || error("Failed to compress!")
    return nothing
end

function SP_d_struct_from_dense(
    S::CSPStructMat,
    A::AbstractMatrix{Float64}, 
    opts::CSPOptions)
    SP_d_struct_from_dense(S.ref, A, opts)
end

function SP_d_struct_from_elements(
    obj::Ref{Ptr{Cvoid}},
    nrows::Integer,
    ncols::Integer,
    func_ptr::Ptr{Cvoid}, 
    opts::CSPOptions)
    success = ccall((:SP_d_struct_from_elements, libstrumpack), 
                    Cint, 
                    (Ref{Ptr{Cvoid}}, Cint, Cint, Ptr{Cvoid}, Ref{CSPOptions}), 
                    obj, nrows, ncols, func_ptr, opts)
    success == 0 || error("Failed to compress!")
    return nothing
end

function SP_d_struct_from_elements(
    S::CSPStructMat,
    nrows::Integer,
    ncols::Integer,
    func_ptr::Ptr{Cvoid}, 
    opts::CSPOptions)
    SP_d_struct_from_elements(S.ref, nrows, ncols, func_ptr, opts)
    return nothing
end

# function SP_d_struct_from_dense2d(
#     S::Ref{Ptr{Cvoid}},
#     comm::MPI.MPI_Comm,
#     A::AbstractMatrix{Float64},
#     DESCA,
#     opts::CSPOptions)
#     m, n = size(A)
#     success = ccall((:SP_d_struct_from_dense2d, libstrumpack), 
#                     Cint, 
#                     (Ref{Ptr{Cvoid}}, MPI.MPI_Comm, Cint, Cint, Ref{Cdouble}, Cint, Cint, Ref{CSPOptions}), 
#                     S, comm, m, n, A, 1, 1, opts)
#     success == 0 || error("Failed to compress!") 
# end

# function SP_d_struct_from_elements_mpi(
#     S::Ref{Ptr{Cvoid}},
#     comm::MPI.Comm,
#     nrows::Integer,
#     ncols::Integer,
#     func_ptr::Ptr{Cvoid},
#     opts::CSPOptions)
#     success = ccall((:SP_d_struct_from_elements_mpi, libstrumpack), 
#                     Cint, 
#                     (Ref{Ptr{Cvoid}}, MPI.MPI_Comm, Cint, Cint, Ptr{Cvoid}, Ref{CSPOptions}), 
#                     S, comm, nrows, ncols, func_ptr, opts)
#     success == 0 || error("Failed to compress!")
#     return nothing
# end

function SP_d_struct_mult(
    obj::Ref{Ptr{Cvoid}}, 
    transp::AbstractChar, 
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64})
    m, n = size(C)
    ccall((:SP_d_struct_mult, libstrumpack), 
          Cint,
          (Ptr{Cvoid}, Cchar, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint),
          obj[], transp, n, B, m, C, m)
end

function SP_d_struct_mult(
    obj::Ref{Ptr{Cvoid}}, 
    transp::AbstractChar, 
    B::AbstractVector{Float64},
    C::AbstractVector{Float64})
    m = length(C)
    ccall((:SP_d_struct_mult, libstrumpack), 
          Cint,
          (Ptr{Cvoid}, Cchar, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint),
          obj[], transp, one(Cint), B, m, C, m)
end

function SP_d_struct_mult(
    S::CSPStructMat, 
    transp::AbstractChar, 
    B::AbstractArray{Float64},
    C::AbstractArray{Float64})
    SP_d_struct_mult(S.ref, transp, B, C)
end

function SP_d_struct_solve(
    obj::Ref{Ptr{Cvoid}},
    B::AbstractMatrix{Float64})
    m, n = size(B)
    ccall((:SP_d_struct_solve, libstrumpack), 
          Cint,
          (Ptr{Cvoid}, Cint, Ref{Cdouble}, Cint),
          obj[], n, B, m)
end

function SP_d_struct_solve(
    obj::Ref{Ptr{Cvoid}},
    b::AbstractVector{Float64})
    m = length(b)
    ccall((:SP_d_struct_solve, libstrumpack), 
          Cint,
          (Ptr{Cvoid}, Cint, Ref{Cdouble}, Cint),
          obj[], one(Cint), b, m)
end

function SP_d_struct_solve(
    S::CSPStructMat,
    B::AbstractArray{Float64})
    SP_d_struct_solve(S.ref, B)
end

function SP_d_struct_destroy(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_destroy, libstrumpack), Cvoid, (Ref{Ptr{Cvoid}},), obj)
end

function SP_d_struct_destroy(S::CSPStructMat)
    if S.ref[] !== C_NULL
        SP_d_struct_destroy(S.ref)
    end
end

function SP_d_struct_factor(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_factor, libstrumpack), Cvoid, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_factor(S::CSPStructMat)
    SP_d_struct_factor(S.ref)
end

function SP_d_struct_rank(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_rank, libstrumpack), Cint, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_rank(S::CSPStructMat)
    SP_d_struct_rank(S.ref)
end

function SP_d_struct_memory(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_memory, libstrumpack), Clonglong, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_memory(S::CSPStructMat)
    SP_d_struct_memory(S.ref)
end

function SP_d_struct_nonzeros(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_nonzeros, libstrumpack), Clonglong, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_nonzeros(S::CSPStructMat)
    SP_d_struct_nonzeros(S.ref)
end

function SP_d_struct_rows(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_rows, libstrumpack), Cint, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_rows(S::CSPStructMat)
    SP_d_struct_rows(S.ref)
end

function SP_d_struct_cols(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_cols, libstrumpack), Cint, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_cols(S::CSPStructMat)
    SP_d_struct_cols(S.ref)
end

function SP_create_kernel_double(
    train::AbstractMatrix{T}, 
    h::T, 
    lambda::T,
    p::Integer,
    type::Integer) where {T<:Float64}
    n, d = size(train)
    ccall((:STRUMPACK_create_kernel_double, libstrumpack),
          Ptr{Cvoid},
          (Cint, Cint, Ref{Cdouble}, Cdouble, Cdouble, Cint, Cint),
          n, d, train, h, lambda, p, type)
end

function SP_kernel_fit_HSS_double(
    kernel::Ptr{Cvoid},
    labels::AbstractVector{Float64},
    argc::Integer,
    argv::Vector{String}
    )
    ccall((:STRUMPACK_kernel_fit_HSS_double, libstrumpack),
          Cvoid,
          (Ptr{Cvoid}, Ref{Cdouble}, Cint, Ptr{Ptr{UInt8}}),
          kernel, labels, argc, argv)
end

function SP_destroy_kernel_double(
    kernel::Ptr{Cvoid}
    )
    ccall((:STRUMPACK_destroy_kernel_double, libstrumpack),
          Cvoid,
          (Ptr{Cvoid},),
          kernel)
end

end