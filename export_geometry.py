import numpy as np
from stl import mesh
import codesign_core as core
from codesign_core import cfg

def export_geometry(c_opt, suffix):
    """Helper function to export CSV and STL for a given thickness field."""
    # Calculate thickness field (convert to mm)
    h_m = np.asarray(core.thickness(c_opt))
    h_mm = h_m * 1e3 
    
    xs_mm = np.linspace(0, cfg.Lx * 1e3, cfg.Nx)
    ys_mm = np.linspace(0, cfg.Ly * 1e3, cfg.Ny)
    X, Y = np.meshgrid(xs_mm, ys_mm, indexing="ij")
    
    # --- LEVEL 1: Export CSV for COMSOL / nTop ---
    csv_filename = f"thickness_field_{suffix}.csv"
    rows = np.column_stack((X.flatten(), Y.flatten(), h_mm.flatten()))
    np.savetxt(csv_filename, rows, 
               header="x_mm,y_mm,thickness_mm", 
               delimiter=",", comments="")
    print(f"  -> Saved {csv_filename}")

    # --- LEVEL 2: Export STL for 3D Printing / CAD ---
    stl_filename = f"optimized_plate_{suffix}.stl"
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
            
    plate_mesh.save(stl_filename)
    print(f"  -> Saved {stl_filename}\n")

def export_results(npz_file="best_params.npz"):
    # Load the optimized parameters
    try:
        data = np.load(npz_file)
    except FileNotFoundError:
        print(f"Error: Could not find {npz_file}. Run your Jupyter notebook first to generate it.")
        return

    found_any = False
    
    # Dynamically determine which mode(s) to load and export
    if 'c_lqr' in data:
        print("Processing LQR mode geometry...")
        export_geometry(data['c_lqr'], "lqr")
        found_any = True
        
    if 'c_str' in data:
        print("Processing Strings mode geometry...")
        export_geometry(data['c_str'], "strings")
        found_any = True
        
    if not found_any:
        print(f"Warning: Neither 'c_lqr' nor 'c_str' found in {npz_file}. Check your save logic.")

if __name__ == "__main__":
    export_results()