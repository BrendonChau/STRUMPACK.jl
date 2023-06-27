module STRUMPACK

using Libdl, LinearAlgebra
import Libdl: dlsym, dlopen

export CSPOptions,
       SP_d_struct_default_options,
       SP_d_struct_from_dense,
       SP_d_struct_mult,
       SP_d_struct_factor,
       SP_d_struct_rank,
       SP_d_struct_destroy

const libstrumpack = "/Users/brendonchau/bin/STRUMPACK-7.1.3/install/lib/libstrumpack.dylib"

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

function SP_d_struct_default_options(opts::CSPOptions)
    ccall((:SP_d_struct_default_options, libstrumpack), Cvoid, (Ref{CSPOptions},), opts)
    # @ccall $(dlsym(libstrumpack, :SP_d_struct_default_options))(opts::Ref{CSPOptions})::Cvoid
end

function SP_d_struct_from_dense(
    S::Ref{Ptr{Cvoid}},
    A::AbstractMatrix{Float64}, 
    opts::CSPOptions)
    m, n = size(A)
    success = ccall((:SP_d_struct_from_dense, libstrumpack), 
                    Cint, 
                    (Ref{Ptr{Cvoid}}, Cint, Cint, Ref{Cdouble}, Cint, Ref{CSPOptions}), 
                    S, m, n, A, m, opts)
    success == 0 || error("Failed to compress!") 
end

function SP_d_struct_mult(
    S::Ref{Ptr{Cvoid}}, 
    T::AbstractChar, 
    B::AbstractArray{Float64},
    C::AbstractArray{Float64})
    m = size(C, 1)
    n = size(C, 2)
    ccall((:SP_d_struct_mult, libstrumpack), Cint,
          (Ptr{Cvoid}, Cchar, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint),
          S[], T, n, B, m, C, m
          )
end

function SP_d_struct_destroy(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_destroy, libstrumpack), Cvoid, (Ref{Ptr{Cvoid}},), obj)
end

function SP_d_struct_factor(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_factor, libstrumpack), Cvoid, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_rank(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_rank, libstrumpack), Cint, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_memory(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_memory, libstrumpack), Clonglong, (Ptr{Cvoid},), obj[])
end

function SP_d_struct_nonzeros(obj::Ref{Ptr{Cvoid}})
    ccall((:SP_d_struct_nonzeros, libstrumpack), Clonglong, (Ptr{Cvoid},), obj[])
end

end