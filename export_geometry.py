import numpy as np
from stl import mesh
import codesign
from codesign import cfg

def export_results(npz_file="best_params.npz"):
    # Load the optimized parameters
    data = np.load(npz_file)
    c_opt = data['c']
    
    # Calculate thickness field (convert to mm)
    h_m = np.asarray(codesign.thickness(c_opt))
    h_mm = h_m * 1e3 
    
    xs_mm = np.linspace(0, cfg.Lx * 1e3, cfg.Nx)
    ys_mm = np.linspace(0, cfg.Ly * 1e3, cfg.Ny)
    X, Y = np.meshgrid(xs_mm, ys_mm, indexing="ij")
    
    # --- LEVEL 1: Export CSV for COMSOL / nTop ---
    rows = np.column_stack((X.flatten(), Y.flatten(), h_mm.flatten()))
    np.savetxt("thickness_field.csv", rows, 
               header="x_mm,y_mm,thickness_mm", 
               delimiter=",", comments="")
    print("Saved thickness_field.csv")

    # --- LEVEL 2: Export STL for 3D Printing / CAD ---
    vertices = []
    faces = []
    Nx, Ny = cfg.Nx, cfg.Ny
    
    # Create Top vertices (Z = h_mm) and Bottom vertices (Z = 0)
    for i in range(Nx):
        for j in range(Ny):
            vertices.append([X[i, j], Y[i, j], h_mm[i, j]])
    for i in range(Nx):
        for j in range(Ny):
            vertices.append([X[i, j], Y[i, j], 0.0])
            
    vertices = np.array(vertices)
    
    def idx(i, j, bottom=False):
        return (Nx * Ny if bottom else 0) + i * Ny + j
        
    # Generate triangular faces for Top and Bottom
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            # Top faces
            faces.append([idx(i,j), idx(i+1,j), idx(i+1,j+1)])
            faces.append([idx(i,j), idx(i+1,j+1), idx(i,j+1)])
            # Bottom faces
            faces.append([idx(i,j,True), idx(i+1,j+1,True), idx(i+1,j,True)])
            faces.append([idx(i,j,True), idx(i,j+1,True), idx(i+1,j+1,True)])
            
    # Create the mesh
    plate_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            plate_mesh.vectors[i][j] = vertices[f[j], :]
            
    plate_mesh.save('optimized_plate.stl')
    print("Saved optimized_plate.stl")

if __name__ == "__main__":
    export_results()