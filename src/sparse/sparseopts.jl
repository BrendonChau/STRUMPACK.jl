function STRUMPACK_set_verbose(
    S::Ref{Ptr{Cvoid}},
    v::Int
    )
    ccall((:STRUMPACK_set_verbose, libstrumpack), 
          Cvoid, (Ptr{Cvoid}, Cint), S[], v)
end

function STRUMPACK_get_verbose(
    S::Ref{Ptr{Cvoid}},
    )
    ccall((:STRUMPACK_verbose, libstrumpack),
            Cint, (Ptr{Cvoid},), S[])
end

function STRUMPACK_set_verbose(
    S::STRUMPACK_Sparse_Solver,
    v::Int
    )
    ccall((:STRUMPACK_set_verbose, libstrumpack), 
          Cvoid, (Ref{STRUMPACK_Sparse_Solver}, Cint), S, v)
end

function STRUMPACK_get_verbose(
    S::STRUMPACK_Sparse_Solver
    )
    ccall((:STRUMPACK_verbose, libstrumpack),
            Cint, (Ref{STRUMPACK_Sparse_Solver},), S)
end

function STRUMPACK_set_maxit(
    S::Ref{Ptr{Cvoid}},
    maxit::Int
    )
    ccall((:STRUMPACK_set_verbose, libstrumpack), 
          Cvoid, (Ptr{Cvoid}, Cint), S[], maxit)
end

function STRUMPACK_get_maxit(
    S::Ref{Ptr{Cvoid}},
    )
    ccall((:STRUMPACK_verbose, libstrumpack),
            Cint, (Ptr{Cvoid},), S[])
end

function STRUMPACK_set_maxit(
    S::STRUMPACK_Sparse_Solver,
    maxit::Int
    )
    ccall((:STRUMPACK_set_verbose, libstrumpack), 
          Cvoid, (Ref{STRUMPACK_Sparse_Solver}, Cint), S, maxit)
end

function STRUMPACK_get_maxit(
    S::STRUMPACK_Sparse_Solver
    )
    ccall((:STRUMPACK_verbose, libstrumpack),
            Cint, (Ref{STRUMPACK_Sparse_Solver},), S)
end

function STRUMPACK_set_matching(
    S::Ref{Ptr{Cvoid}},
    job::STRUMPACK_MATCHING_JOB
    )
    ccall((:STRUMPACK_set_matching, libstrumpack),
            Cvoid, (Ptr{Cvoid}, Cint), S[], job)
end

function STRUMPACK_set_reordering_method(
    S::Ref{Ptr{Cvoid}},
    m::STRUMPACK_REORDERING_STRATEGY
    )
    ccall((:STRUMPACK_set_reordering_method, libstrumpack),
            Cvoid, (Ptr{Cvoid}, Cint), S[], m)
end

function STRUMPACK_set_compression(
    S::Ref{Ptr{Cvoid}},
    t::STRUMPACK_COMPRESSION_TYPE
    )
    ccall((:STRUMPACK_set_compression, libstrumpack),
            Cvoid,
            (Ptr{Cvoid}, Cint),
            S[], t)
end

function STRUMPACK_set_compression_leaf_size(
    S::Ref{Ptr{Cvoid}},
    leafsize::Int
    )
    ccall((:STRUMPACK_set_compression_leaf_size, libstrumpack),
            Cvoid,
            (Ptr{Cvoid}, Cint),
            S[], leafsize)
end

function STRUMPACK_set_compression_rel_tol(
    S::Ref{Ptr{Cvoid}},
    rctol::Cdouble
    )
    ccall((:STRUMPACK_set_compression_rel_tol, libstrumpack),
            Cvoid,
            (Ptr{Cvoid}, Cdouble),
            S[], rctol)
end

function STRUMPACK_set_compression_min_sep_size(
    S::Ref{Ptr{Cvoid}},
    size::Int
    )
    ccall((:STRUMPACK_set_compression_min_sep_size, libstrumpack),
            Cvoid,
            (Ptr{Cvoid}, Cint),
            S[], size)
end