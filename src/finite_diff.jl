using SparseArrays
using JLD2
using FileIO
function mod_idx(idx,n)
    if idx < 1
        return idx + n
    elseif idx > n
        return idx - n
    else 
        return idx
    end
end

function diff_mat2d(nx,ny,along,odiff,oacc=4,load_from_file =true)
    """
    nx,ny : the number of points in each dimension
    along : derivative along x-axis(along=1) or y-axis(along=2)
    odiff : order of the derivative
    oacc  : order of accuracy
    load_from_file : try to load precaluclated matrices
    return: the difference matrix
    *note that the marix if for periodic boundary. 
    *TBD : we introduce ghost layers for other boundary conditions to save the labour of 
        modifying the difference matrices, and these layers should be reset after 
        each iteration in time
    """
    if load_from_file
        file_name = "diff_mat2d_nx$(nx)_ny$(ny)_along$(along)_odiff$(odiff)_oacc$(oacc).jld2"
        # @__FILE__ is the location for the current running file
        # dirname(LOC) get the parent directory of LOC
        directory = joinpath(dirname(dirname(@__FILE__)),"save_diff_mat")
        full_file_name = joinpath(directory,file_name)
        if isfile(full_file_name)
            diff_mat = JLD2.load(full_file_name,"diff_mat2d")
            return diff_mat
        end
    end
        
    N = nx*ny
    if oacc == 2
        cdiff_coeff = [0.0  -0.5  0.0   0.5   0.0;
                       0.0   1.0 -2.0   1.0   0.0]
    elseif oacc == 4
        cdiff_coeff = [1.0/12.0  -2.0/3.0  0.0      2.0/3.0  -1.0/12.0;
                      -1.0/12.0   4.0/3.0 -5.0/2.0 	4.0/3.0  -1.0/12.0]
    elseif oacc == 6
        cdiff_coeff = [-1.0/60.0  3.0/20.0  -3.0/4.0  0.0        3.0/4.0  -3.0/20.0  1.0/60.0;
                        1.0/90.0 -3.0/20.0 	 3.0/2.0 -49.0/18.0  3.0/2.0  -3.0/20.0  1.0/90.0]
    elseif oacc == 8
        cdiff_coeff = [1.0/280.0  -4.0/105.0  1.0/5.0  -4.0/5.0   0.0         4.0/5.0  -1.0/5.0  4.0/105.0  -1.0/280.0;
                      -1.0/560.0   8.0/315.0 -1.0/5.0   8.0/5.0  -205.0/72.0  8.0/5.0  -1.0/5.0  8.0/315.0  -1.0/560.0]
    else
        error("oacc = 2,4,6 or 8, no other values allowed")
    end
    nrow,ncol = size(cdiff_coeff)
    nbrs = ncol ÷ 2         # integer division
    diff_mat = spzeros(Float64, N, N)
    stencil_ind =  Array{Int64,1}(undef,ncol)

    #=
    # a special case for order of accuracy = 2
    for i in 1:nx
        for j in 1:ny   
            if along == 1 
                stencil_ind[3] = i + (j-1)*nx                  # center
                stencil_ind[2] = mod_idx(i - 1, nx) + (j-1)*nx # left one
                stencil_ind[1] = mod_idx(i - 2, nx) + (j-1)*nx # left two
                stencil_ind[4] = mod_idx(i + 1, nx) + (j-1)*nx # right one
                stencil_ind[5] = mod_idx(i + 2, nx) + (j-1)*nx # right two
            elseif along == 2
                stencil_ind[3] = i + (j-1)*nx                        # center
                stencil_ind[2] = i + (mod_idx(j - 1, ny) - 1)*nx     # upper one
                stencil_ind[1] = i + (mod_idx(j - 2, ny) - 1)*nx     # upper two
                stencil_ind[4] = i + (mod_idx(j + 1, ny) - 1)*nx     # lower one
                stencil_ind[5] = i + (mod_idx(j + 2, ny) - 1)*nx     # lower two
            else
                error("along = 1 or 2, no other values allowed")
            end
            for is in 1:5
                diff_mat[stencil_ind[3],stencil_ind[is]] = cdiff_coeff[odiff,is]
            end
        end
    end
    =#
    for i in 1:nx
        for j in 1:ny   
            if along == 1 
                stencil_ind[nbrs + 1] = i + (j-1)*nx                             # center
                for nl in 1:nbrs
                    stencil_ind[nbrs + 1 - nl] = mod_idx(i - nl, nx) + (j-1)*nx   # the point left to the center by nl unit
                end
                for nr in 1:nbrs
                    stencil_ind[nbrs + 1 + nr] = mod_idx(i + nr, nx) + (j-1)*nx  # right one
                end
            elseif along == 2
                stencil_ind[nbrs + 1] = i + (j-1)*nx                              # center
                for nu in 1:nbrs
                    stencil_ind[nbrs + 1 - nu] = i + (mod_idx(j - nu, ny) - 1)*nx  # the point up to the center by nu unit
                end
                for nd in 1:nbrs
                    stencil_ind[nbrs + 1 + nd] = i + (mod_idx(j + nd, ny) - 1)*nx # the point down to the center by nd unit
                end
            else
                error("along = 1 or 2, no other values allowed")
            end 
            for is in 1:ncol
                diff_mat[stencil_ind[nbrs + 1],stencil_ind[is]] = cdiff_coeff[odiff,is]
            end
        end
    end
    dropzeros!(diff_mat)
    if load_from_file
        # save the calculated matix for future use
        file_name = "diff_mat2d_nx_$(nx)_ny$(ny)_along$(along)_odiff$(odiff)_oacc$(oacc).jld2"
        # @__FILE__ is the location for the current running file
        # dirname(LOC) get the parent directory of LOC
        directory = joinpath(dirname(dirname(@__FILE__)),"save_diff_mat")
        full_file_name = joinpath(directory,file_name)
        if isdir(directory)
            FileIO.save(full_file_name,"diff_mat2d",diff_mat)
        else
            mkdir(directory)
            JLD2.save(full_file_name,"diff_mat2d",diff_mat)
        end
    end
    return sparse(diff_mat)
end

function mixed_diff_mat2d(mdiff::Tuple{Integer,Integer},nx,ny,dx,dy)
    if mdiff[1] == 0
        diff_x = 1.0/(dx^mdiff[1])
    else
        diff_x = diff_mat2d(nx,ny,1,mdiff[1])/(dx^mdiff[1])
    end

    if mdiff[2] == 0
        diff_y = 1.0/(dy^mdiff[2])
    else
        diff_y = diff_mat2d(nx,ny,2,mdiff[2])/(dy^mdiff[2])
    end
    return sparse(diff_x*diff_y)
end


function cut_into_blocks(M::SparseMatrixCSC{Float64, Int64},n)
"""
    cut the Nx-by-Ny sparse matrix,M, into n matrices, B1,B2 ... Bn,
    where B1 = M[1:s,:] B2 = M[s+1:2s,:],Bi = M[(i-1)*s+1:i*s,:]
    s = div(Nx,n)
"""
    Nx,Ny = size(M)
    s = div(Nx,n)
    diff_mat_list = Array{SparseMatrixCSC{Float64, Int64},1}(undef,n) 
    for i in 1:n 
        diff_mat_list[i] = M[(i-1)*s+1:i*s,:]
    end
    return diff_mat_list
end

function simpson_int1d(f::Array{Float64,1},rng)
"""
    1d numerical integration using Simpson's rule
    f is the integrand of N+1 points
    rng contains the limits, a and b, of the integration
    f[i] is the value of f(a + (i-1)*h), i = 1...N+1 
    where h = (b-a)/N 
"""
    if ndims(f) > 1
        error("ndims(f) must be 1 for 1d numerical integration")
    end
    N = size(f)[1] - 1
    h = (rng[2] - rng[1])/N
    s = f[1] + f[N+1]
    for j in 2:2:N
        s += 4.0*f[j]
    end
    for j in 3:2:N
        s += 2.0*f[j]
    end
    return s*h/3.0
end

function simpson_int2d(f::Array{Float64,2},rng_x,rng_y)
"""
    2d numerical integration using Simpson's rule
    f     :the integrand of (Nx+1)*(Ny+1) points
    rng_x :the limit for x direction, xi and xf
    rng_y :the limit for y direction, yi and yf
    f[i,j] is the value of f(xi + (i-1)*hx, yi + (j-1)*hy)
           , i = 1...Nx+1 j = 1...Ny+1
    where hx = (xf-xi)/Nx and hy = (yf-yi)/Ny 
"""
    Nx,Ny = size(f) .- 1
    # the auxiliary varible fy = int dx f(x,y)
    fy = zeros(Float64,Ny + 1)
    for y in 1:Ny+1
        fy[y] = simpson_int1d(f[:,y],rng_x)
    end
    return simpson_int1d(fy,rng_y)
end

function get_by_pos(nx,ny,M::Array{Float64,2})
"""
   lattice position to array indicies in a 2Nx+1-by-2Ny+1 matrix M
   lattice position (nx,ny)=(0,0) corresponding to M[Nx+1,Ny+1]
   lattice position (nx,ny)=(p,q) corresponding to M[p+Nx+1,q+Ny+1]
"""
   Nx,Ny = size(M)
   Nx = div(Nx,2)
   Ny = div(Ny,2)
   return M[nx+Nx+1,ny+Ny+1]
end

function set_by_pos!(nx,ny,M,val)
    """
       lattice position to array indicies in a 2N+1-by-2N+1 matrix M
       lattice position (nx,ny)=(0,0) corresponding to M[N+1,N+1]
       lattice position (nx,ny)=(i,j) corresponding to M[i+N+1,j+N+1]
    """
    Nx,Ny = size(M)
    Nx = div(Nx,2)
    Ny = div(Ny,2)
    M[nx+Nx+1,ny+Ny+1] = val
end

function corr2d(ker::Array{Float64,2},g::Array{Float64,2},c,dx,dy)
"""
    calculate the 2d correlation function at point c (2d point)
    c:  represent the point (c[1]*dx,c[2]*dy)
    corr = int dx2 k(x1-x2)g(x2) on 
           x2 in [(c[1]-N)*dx,(c[1]+N)*dx] * [(c[2]-N)*dy,(c[2]+N)*dy] 
    ker: (2N+1)*(2N+1) kernel matrix with 
        ker represent the point of a continous function K(r) centered at zero
        with by_pos(i,j,ker) = Ker(i*dx,j*dy) 
    g: a lattice representation of the periodic function G(x,y)
       g[i,j] = G(i*dx,j*dy)
"""
    Nx,Ny = size(ker)
    nx,ny = size(g)
    Nx,Ny = div(Nx,2),div(Ny,2)
    # I is an intermediate vaiable for the integrand
    # I represent the continuous function Integrand
    # where Integrand(i*dx,j*dy) = I[i-c[1],(j-c[2]]
    # Integrand(i*dx,j*dx) = Ker((c[1]-i)*dx, (c[2]-j)*dy) * G(i*dx,j*dy) 
    # this implies
    # I[i-c[1],(j-c[2]]] = get_by_pos(c[1]-i,c[2]-j,ker)*g[i,j]
    I = zeros(Float64,2*Nx+1,2*Ny+1)
    for i in c[1]-Nx:c[1]+Nx
        for j in c[2]-Ny:c[2]+Ny
            _i = mod_idx(i,nx)
            _j = mod_idx(i,ny)
            val =  get_by_pos(c[1]-i, c[2]-j,ker)*g[_i,_j]
            set_by_pos!(i-c[1],j-c[2],I,val)
        end
    end
    return simpson_int2d(I,((c[1]-Nx)*dx,(c[1]+Nx)*dx), ((c[2]-Ny)*dy,(c[2]+Ny)*dy))
end
