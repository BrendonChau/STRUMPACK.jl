using Random, LinearAlgebra, StatsBase, BenchmarkTools, SparseArrays
using Plots
using STRUMPACK
using SparseMatricesCSR

ENV["OMP_NUM_THREADS"] = 1
BLAS.set_num_threads(1)

Plots.default(; size = (1200, 800))

rng = Random.MersenneTwister(123)

n = 512
d = 1024
maxiter = 8
q = d * maxiter

M = Matrix{Float64}(undef, n, n)
X = Matrix{Float64}(undef, n, d)

function scale!(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for j in axes(A, 2)
        ctr = mean(view(A, :, j))
        scal = std(view(A, :, j))
        view(A, :, j) .-= ctr
        view(A, :, j) ./= scal
    end
    return A
end

function update_M!(
    rng::Random.AbstractRNG, 
    M::AbstractMatrix{T}, 
    buffer::AbstractMatrix{T}, 
    maxiter::Integer
    ) where {T<:AbstractFloat}
    fill!(M, zero(T))
    denom = size(buffer, 2) * maxiter
    @inbounds for _ in 1:maxiter
        randn!(rng, buffer)
        scale!(buffer)
        BLAS.syrk!('L', 'N', inv(denom), X, one(T), M)
    end
    LinearAlgebra.copytri!(M, 'L')
end

update_M!(rng, M, X, maxiter)

# function fill_toeplitz!(M::AbstractMatrix{T}) where {T<:AbstractFloat}
#     n = size(M, 1)
#     @inbounds for j in axes(M, 2)
#         for i in j:n
#             M[i, j] = inv(one(T) + abs(i - j))
#         end
#     end
#     LinearAlgebra.copytri!(M, 'L')
# end

# fill_toeplitz!(M)

# function _getindex(A::AbstractMatrix{Float64}, i::Tint, j::Tint)::Cdouble where {Tint<:Integer}
#     @inbounds Base.getindex(A, i + 1, j + 1)
# end

# _getindex_c = @cfunction((i, j) -> _getindex(M, i, j), Cdouble, (Cint, Cint))

# @inline function toeplitz(i::Tint, j::Tint)::Cdouble where {Tint<:Integer}
#     inv(one(Cdouble) + abs(i - j))
# end
# toeplitz_c = @cfunction(toeplitz, Cdouble, (Cint, Cint))

begin
    opts = CSPOptions()

    SP_d_struct_default_options(opts)

    opts.type = STRUMPACK.SP_TYPE_HSS
    opts.rel_tol = 1e-4
    opts.abs_tol = 1e-6
    opts.leaf_size = 128
    opts.max_rank = 512
    opts.verbose = 1
end

# Xt = zeros(d, n)
# transpose!(Xt, X)
# D = Matrix{Float64}(undef, n, n)
# function euclidean_dist_squared!(
#     K::Matrix{T}, 
#     X_t::Matrix{T}
#     ) where {T<:AbstractFloat}
#     @inbounds for k in axes(K, 2)
#         for j in axes(K, 1)
#             K[j, k] = zero(T)
#             for i in axes(X_t, 1)
#                 K[j, k] += abs2(X_t[i, j] - X_t[i, k])
#             end
#         end
#     end
# end

# euclidean_dist_squared!(D, Xt)

# K = exp.(-D / (2*q))

# [svdvals(K) svdvals(M)]

# argv = ["--hss_verbose=true", 
#         "--hss_ann_iterations=5", # Only called with Kernel matrices
#         "--hss_approximate_neighbors=128",
#         "--hss_clustering_algorithm=2means", 
#         "--hss_compression_algorithm=stable",
#         "--hss_random_distribution=normal",
#         "--hss_d0=128",
#         "--hss_dd=64"]
#         # "--hss_compression_sketch=SJLT",
#         # "--hss_SJLT_algo=chunk",
#         # "--hss_nnz0=8",
#         # "--hss_nnz=8"]

# # argv = ["--blr_verbose=true", 
# #         "--blr_low_rank_algorithm=ACA",
# #         "--blr_admissibility=weak",
# #         "--blr_compression_kernel=full"]

# @show S = STRUMPACK.CSPStructMat()

# @timev SP_d_struct_from_dense(S, M, opts, argv)
# # @timev SP_d_struct_from_elements(S, n, n, toeplitz_c, opts)

# @show SP_d_struct_rank(S)

# @show STRUMPACK.SP_d_struct_nonzeros(S)

# @show STRUMPACK.SP_d_struct_memory(S) / 1e6

# @show sizeof(M) / 1e6

# # b = ones(n);
# b = randn(rng, n);
# c_approx = similar(b);
# c_exact  = similar(b);

# @timev SP_d_struct_mult(S, 'N', b, c_approx)

# # @timev BLAS.symv!('L', 1.0, M, b, 0.0, c_exact);
# # @timev BLAS.symm!('L', 'L', 1.0, M, b, 0.0, c_exact);
# @timev BLAS.gemv!('N', 1.0, M, b, 0.0, c_exact);

# println()
# @show norm(c_approx - c_exact)

# M_approx = Matrix{Float64}(undef, n, n);
# I_n = Matrix{Float64}(UniformScaling(1.0), n, n);
# SP_d_struct_mult(S, 'N', I_n, M_approx)

# @show norm(M - M_approx) / norm(M)

# @timev STRUMPACK.SP_d_struct_factor(S)

# # @timev LAPACK.potrf!('L', M);

# STRUMPACK.SP_d_struct_destroy(S)

# Sparse Matrix Examples
M_ht = Matrix{Float64}(undef, n, n);

for j in 1:n
    for i in j:n
        if i == j
            M_ht[i, j] = M[i, j]
        else
            M_ht[i, j] = ifelse(abs(M[i, j]) < 0.0112, 0.0, M[i, j])
        end
    end
end
LinearAlgebra.copytri!(M_ht, 'L')

M_sp = sparse(M_ht)

rowptr  = Int32.(M_sp.colptr)
col_ind = Int32.(M_sp.rowval)
values = M_sp.nzval

sp_S = STRUMPACK.STRUMPACK_Sparse_Solver()
# @show sp_S.solver

STRUMPACK.STRUMPACK_init_mt(sp_S,
                            STRUMPACK.STRUMPACK_DOUBLE,
                            STRUMPACK.STRUMPACK_MT,
                            ["--help",
                             "--sp_verbose"],
                            1)

@show STRUMPACK.STRUMPACK_get_verbose(sp_S)

@show STRUMPACK.STRUMPACK_set_verbose(sp_S, 1)

@show STRUMPACK.STRUMPACK_get_verbose(sp_S)

# structinfo(T) = [(fieldoffset(T,i), fieldname(T,i), fieldtype(T,i)) for i = 1:fieldcount(T)];
# @show structinfo(STRUMPACK.STRUMPACK_Sparse_Solver)

@show sp_S.solver
@show sp_S.precision
@show sp_S.interface

# STRUMPACK.STRUMPACK_set_matching(sp_S, STRUMPACK.STRUMPACK_MATCHING_NONE)
# STRUMPACK.STRUMPACK_set_reordering_method(sp_S, STRUMPACK.STRUMPACK_GEOMETRIC)
# STRUMPACK.STRUMPACK_set_compression(sp_S, STRUMPACK.STRUMPACK_BLR)

# STRUMPACK.STRUMPACK_set_compression_leaf_size(sp_S, 64)
# STRUMPACK.STRUMPACK_set_compression_rel_tol(sp_S, 1e-2) 
# STRUMPACK.STRUMPACK_set_compression_min_sep_size(sp_S, 256)

# STRUMPACK.STRUMPACK_set_maxit(sp_S, 100)
# STRUMPACK.STRUMPACK_get_maxit(sp_S)

STRUMPACK.STRUMPACK_set_csr_matrix(sp_S, n, rowptr, col_ind, values, 1)

# @show STRUMPACK.STRUMPACK_reorder_regular(sp_S, n, n, 1, 1, 1)

# @show STRUMPACK.STRUMPACK_factor(sp_S)

# b = rand(rng, n);
# x = Vector{Float64}(undef, n);
# fill!(x, 0)

# @show STRUMPACK.STRUMPACK_solve(sp_S, b, x, 0)

# b
# x

STRUMPACK.STRUMPACK_destroy(sp_S)

# # Kernel Matrix Approximation
# p = 1024 * 4;
# Z = Matrix{Float64}(undef, n, p);
# randn!(rng, Z)
# scale!(Z)

# Z ./= sqrt(p)

# BLAS.syrk!('L', 'N', 1.0, Z, 0.0, M)
# LinearAlgebra.copytri!(M, 'L')

# println("Dense Kernel Matrix size in Mb:")
# @show (abs2(n) * sizeof(Float64)) / 1e6

# G = STRUMPACK.SP_create_kernel_double(Z, 1.0, 0.0001, 1, 0)

# # G = STRUMPACK.SP_create_kernel_double(Z, M, 1e-4)

# y = randn(rng, n);

# argv = ["--hss_verbose=true", 
#         "--hss_ann_iterations=100", 
#         "--hss_clustering_algorithm=cobble", 
#         "--hss_rel_tol=0.01",
#         "--hss_abs_tol=0.10",
#         "--hss_approximate_neighbors=128",
#         "--hss_d0=128",
#         "--hss_dd=128",
#         "--hss_max_rank=512",
#         "--hss_leaf_size=128",
#         "--hss_compression_sketch=SJLT",
#         "--hss_SJLT_algo=chunk",
#         "--hss_nnz0=8",
#         "--hss_nnz=8"]

# STRUMPACK.SP_kernel_fit_HSS_double(G, y, argv)

# STRUMPACK.SP_destroy_kernel_double(G);
