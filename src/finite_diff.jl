using SparseArrays
function diff_mat2d(nx,ny,along,odiff)
    """
    nx,ny : the number of points in each dimension
    along : derivative along x-axis(along=1) or y-axis(along=2)
    odiff : order of the derivative
    return: the difference matrix
    *note that the marix if for periodic boundary. 
    *TBD : we introduce ghost layers for other boundary conditions to save the labour of 
        modifying the difference matrices, and these layers should be reset after 
        each iteration in time
    """
    function mod_idx(idx,n)
        if idx < 1
            return idx + n
        elseif idx > n
            return idx - n
        else 
            return idx
        end
    end
        
    N = nx*ny
    cdiff_coeff = [0.0  -0.5  0.0   0.5   0.0;
                   0.0   1.0 -2.0   1.0   0.0;
                  -0.5   1.0  0.0  -1.0   0.5;
                   1.0  -4.0  6.0  -4.0   1.0]
    diff_mat = spzeros(Float64, N, N)
    stencil_ind =  Array{Int64,1}(undef,5)
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
    dropzeros!(diff_mat)
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