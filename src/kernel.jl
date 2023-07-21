function SP_create_kernel_double(
    train::AbstractMatrix{T}, 
    h::T, 
    lambda::T,
    p::Integer, # Only matters for ANOVA kernel
    type::Integer) where {T<:Float64}
    n, d = size(train)
    ccall((:STRUMPACK_create_kernel_double, libstrumpack),
          Ptr{Cvoid},
          (Cint, Cint, Ref{Cdouble}, Cdouble, Cdouble, Cint, Cint),
          n, d, train, h, lambda, p, type)
end

function SP_create_kernel_double(
    train::AbstractMatrix{T}, 
    dense::Matrix{T},
    lambda::T
    ) where {T<:Float64}
    n, d = size(train)
    ccall((:STRUMPACK_create_dense_kernel_double, libstrumpack),
          Ptr{Cvoid},
          (Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cdouble),
          n, d, train, dense, lambda)
end

function SP_kernel_fit_HSS_double(
    kernel::Ptr{Cvoid},
    labels::AbstractVector{Float64},
    argv::Vector{String}
    )
    argc = length(argv)
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