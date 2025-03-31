import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
import numpy as np
from qutip import *
from qutip.piqs import *
from qutip import piqs
import PhaseSpaceRepresentations_upgr as ps

#import matplotlib.animation as animation
#from IPython.display import HTML
#from IPython.core.display import Image, display


def rotation_y(rho,j_y,N,phi):
    system = piqs.Dicke(N = N)
    system.hamiltonian = 0.5*phi*j_y
    D_tls = system.liouvillian()
    t = np.linspace(0, 1, 100)
    
    result = mesolve(D_tls, rho, t, [])
    return result.states[-1]

def OAT(rho,j_z,N,Q_per_nsc,F_per_nsc,n_sc,scattering=True):
    
    system = piqs.Dicke(N = N)
    t = np.linspace(0, 1, 100) #tau=1
    chi=Q_per_nsc*n_sc/N
    system.hamiltonian = chi*j_z**2
    if scattering:
        system.dephasing =2*n_sc/N
    system.collective_dephasing=F_per_nsc*n_sc/N
    
    D_tls = system.liouvillian()
    
    result = mesolve(D_tls, rho, t, [])
    return result.states[-1]

def density_op_plot_grid(ax,N):

    lines=[N]
    if N%2==0:
        for S in np.flip(0.5*np.arange(0,N,2)):
            ax.axhline(lines[-1],c='black')
            ax.axvline(lines[-1],c='black')
            lines.append(lines[-1]+2*S+1)    
    else:
        for S in np.flip(0.5*np.arange(1,N,2)):
            ax.axhline(lines[-1],c='black')
            ax.axvline(lines[-1],c='black')
            lines.append(lines[-1]+2*S+1)

# def isolate_density_op_block(rho,N,S):
#     index=0
#     for s in np.flip(0.5*np.arange(2*S,N+2,2))[:-1]:
#         index=index+2*s+1
#     return rho[int(index):int(index+2*S+1),int(index):int(index+2*S+1)]

def isolate_density_op_block(rho, N, S):
    """
    Extract the block corresponding to total spin S from the density matrix.
    
    Parameters:
    rho: density matrix
    N: number of spins (N=20 in your case)
    S: desired total spin quantum number
    
    Returns:
    block: (2S+1)×(2S+1) density matrix block for spin S
    """
    # Check trace of full density matrix using QuTiP's trace
    print(f"Shape of rho: {rho.shape}")
    print(f"Trace of rho: {(rho.tr()).real}")
    # Start from maximum S (N/2) and sum dimensions of all blocks until we reach our desired S
    S_max = N/2
    index = 0
    
    # Sum dimensions of all blocks with higher S values
    for S_block in np.arange(S_max, S, -1):
        index += int(2*S_block + 1)
    
    block_size = int(2*S + 1)
    block = rho[index:index+block_size, index:index+block_size]
    
    # Add verification
    print(f"For S={S}:")
    print(f"  Block starts at index {index}")
    print(f"  Block size: {block_size}×{block_size}")
    print(f"  Block trace: {np.real(np.trace(block)):.6f}")
    print(f"  Trace Block^2: {np.real(np.trace(block@block))}")
    
    return block

###Initialize system
N = 20
iters = 1
system = piqs.Dicke(N = N)
[jx, jy, jz] = piqs.jspin(N)
jp = piqs.jspin(N,"+")
jm = jp.dag()


#OAT dynamics with dephasing:
Q_per_nsc=30#60#0.23 #for N=200
F_per_nsc=0.#9#0.06 #for N=200
n_sc=0.7#0.7#0.1
scattering_rates = np.arange(0, 1.1, 0.1)

most_negative_results = {}  # Dictionary to store results for each scattering rate

for n_sc in scattering_rates:
    print(f"\nAnalyzing scattering rate n_sc = {n_sc:.1f}")

    ##Fully polarized state to begin with
    rho0_tls = piqs.dicke(N, N/2, N/2)

    #Bring to equator of Bloch sphere
    rhot_tls = rotation_y(rho0_tls,jy,N,np.pi) #result.states

    #Oat dynamics with dephasing
    for i in range(iters):
        rhot_tls = OAT(rhot_tls,jz,N,Q_per_nsc,F_per_nsc,n_sc,scattering=True)
        rhot_tls=rotation_y(rhot_tls,jy,N,np.pi/2)

    ###FAST PLOTTING (pick which Bloch sphere to use)
    path2kernels = 'Calculated/Kernels/'
    negative_sums = {}
    total_p = 0

    for S in np.flip(np.arange(1,int(N/2)+1)): ##automate better, for larger N

        Ndim=int(2*S+1)

        rho=isolate_density_op_block(rhot_tls,N,S)
        p=np.trace(rho)
        p=np.real(p)
        total_p += p
        #rho=rho/p


        Kcoeffs = ps.precalculatedKcoeffs(Ndim, path2kernels)
        finalpoints=int(256*1.5)
        result = ps.PSrepresentationFromFourier(rho, Kcoeffs, finalpoints)

        # Calculate sum of negative values
        negative_sum = np.sum(result[result < 0])
        negative_sums[S] = negative_sum
        
        print(f"S={S}: Sum of negative values = {negative_sum:.6f}")

        # Create directory for this scattering rate if it doesn't exist
        scatter_dir = f'subspace_results/n_sc_{n_sc:.1f}'
        if not os.path.exists(scatter_dir):
            os.makedirs(scatter_dir)

        filename = f'{scatter_dir}/S_{S}'
        ps.PSrepPlot_plane_save(result,f'S={S}, p={p:.3f}, n_sc={n_sc:.1f}',filename)

    # Find S with most negative sum for this scattering rate
    most_negative_S = min(negative_sums.items(), key=lambda x: x[1])[0]
    most_negative_value = negative_sums[most_negative_S]
    most_negative_results[n_sc] = {
        'S': most_negative_S,
        'value': most_negative_value
    }
    print(f"Most negative contributions at n_sc={n_sc:.1f}: S={most_negative_S}, sum={most_negative_value:.6f}")
    
    # Print total probability after loop
    print(f"\nTotal probability across all S values: {total_p:.6f}")
    if abs(total_p - 1.0) > 1e-6:  # Check if significantly different from 1
        print("WARNING: Total probability is not 1!")   
# Save summary of results
with open('negative_values_summary.txt', 'w') as f:
    f.write("Summary of most negative contributions for each scattering rate:\n")
    f.write("------------------------------------------------\n")
    for n_sc, result in most_negative_results.items():
        f.write(f"n_sc = {n_sc:.1f}:\n")
        f.write(f"  Most negative S = {result['S']}\n")
        f.write(f"  Negative sum = {result['value']:.6f}\n")
        f.write("------------------------------------------------\n")

# Create a plot of most negative values vs scattering rate
plt.figure(figsize=(10, 6))
scatter_rates = list(most_negative_results.keys())
negative_values = [result['value'] for result in most_negative_results.values()]
plt.plot(scatter_rates, negative_values, 'bo-')
plt.xlabel('Scattering Rate (n_sc)')
plt.ylabel('Sum of Negative Values')
plt.title('Most Negative Contributions vs Scattering Rate')
plt.grid(True)
plt.savefig('negative_values_vs_scattering.png')
plt.close()