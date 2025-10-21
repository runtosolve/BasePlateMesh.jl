# Activate Environment Folder : base_plate_modeling_2025
using PyCall

math = pyimport("math")
math.sin(math.pi / 4) # returns ≈ 1/√2 = 0.70710678...


gmsh = pyimport("gmsh")

# using Ferrite
# using GLMakie
# using Tensors
# using FerriteGmsh
# using GeometryBasics
# using LinearAlgebra
const gmsh = pyimport_conda("gmsh", "gmsh")
# using ForwardDiff 
# using StaticArrays

mesh_filename = "baseplate.msh"

gmsh.initialize()
gmsh.model.add("BasePlate")

# Plate dimensions
L, W, t = 300.0, 300.0, 10.0

# hole radius
dia = 20.0
r = dia*0.5
element_type = "tri"   # "quad" for quadrilateral or "tri" for triangular, "tetra" for tetrahedron, "hexa" for hexahedron
model_type = "2D"       # "2D" or "3D"


box_half = 1.0 * dia
hole_locations = [(-100.0, -100.0), (100.0, -100.0), (-100.0, 100.0), (100.0, 100.0)]

if model_type == "2D"
    plate = gmsh.model.occ.addRectangle(-L/2, -W/2, 0.0, L, W)
elseif model_type == "3D"
    plate = gmsh.model.occ.addBox(-L/2, -W/2, 0.0, L, W, t)
end

#reference: https://gmsh.info/doc/texinfo/gmsh.html

# **OpenCASCADE (OCC) geometry kernel**
# baseplate = gmsh.model.occ.cut([(2, plate)], [(2, h) for h in holes]; removeObject = true, removeTool = true)[1]

# Note:
# in (3, plate), the number indicates the type of the geometric entity. In Gmsh, dimensions are:
# 0: Points
# 1: Lines/Curves
# 2: Surfaces
# 3: Volumes (3D objects)
# Tutorial https://jsdokken.com/src/tutorial_gmsh.html

# Add square refinement boxes
boxes = []
for (x, y) in hole_locations
    xmin, ymin = x - box_half, y - box_half
    if model_type == "2D"
        box = gmsh.model.occ.addRectangle(xmin, ymin, 0.0, 2 * box_half, 2 * box_half)
    elseif model_type == "3D"
        box = gmsh.model.occ.addBox(xmin, ymin, 0.0, 2 * box_half, 2 * box_half, t)
    end
    push!(boxes, box)
end

if model_type == "2D"
    fragmented_plate = gmsh.model.occ.fragment([(2, plate)], [(2, b) for b in boxes])[1]
    holes = [gmsh.model.occ.addDisk(x,y,0.0,r,r) for (x,y) in hole_locations]
    final_plate = gmsh.model.occ.cut(fragmented_plate, [(2, h) for h in holes];removeObject=true, removeTool=true)[1]

elseif model_type == "3D"
    fragmented_plate = gmsh.model.occ.fragment([(3, plate)], [(3, b) for b in boxes])[1]
    holes = [gmsh.model.occ.addCylinder(x,y,0.0,0.0,0.0,t,r) for (x,y) in hole_locations]
    final_plate = gmsh.model.occ.cut(fragmented_plate, [(3, h) for h in holes];removeObject=true, removeTool=true)[1]
end


# final_plate = gmsh.model.occ.cut([(2, plate)], [(2, h) for h in holes]; removeObject = true, removeTool = true)[1]
# final_plate = gmsh.model.occ.cut(fragmented_plate, [(2, h) for h in holes];removeObject=true, removeTool=true)[1]

gmsh.model.occ.synchronize()


# Mesh size definition
uniform_mesh_size = 5
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", uniform_mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", uniform_mesh_size)

# Generate Mesh Model
if element_type in ("quad", "hexa")
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
elseif element_type == ("tri", "tetra")
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
end 

if model_type == "2D"
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_filename)
    gmsh.finalize()
    grid = FerriteGmsh.togrid(mesh_filename)
elseif model_type == "3D"
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_filename)
    gmsh.finalize()
    grid = FerriteGmsh.togrid(mesh_filename)
end


if model_type == "2D"
    if element_type == "quad"
        mesh_faces = QuadFace[]
        for cell in grid.cells
            nodes = collect(cell.nodes)
            push!(mesh_faces, QuadFace(nodes[1], nodes[2], nodes[3], nodes[4]))
        end
    elseif element_type == "tri"
        mesh_faces = GLTriangleFace[]
        for cell in grid.cells
            nodes = collect(cell.nodes)
            push!(mesh_faces, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
        end
    end

elseif model_type == "3D"
    if element_type == "tetra"
        # mesh_faces = GLTriangleFace[]
        mesh_faces = GLTriangleFace[]
        for cell in grid.cells
            cell_node_indices = collect(cell.nodes)   
            push!(mesh_faces, GLTriangleFace(cell_node_indices[1], cell_node_indices[2], cell_node_indices[3]))
            push!(mesh_faces, GLTriangleFace(cell_node_indices[1], cell_node_indices[2], cell_node_indices[4]))
            push!(mesh_faces, GLTriangleFace(cell_node_indices[1], cell_node_indices[3], cell_node_indices[4]))
            push!(mesh_faces, GLTriangleFace(cell_node_indices[2], cell_node_indices[3], cell_node_indices[4]))
        end

    elseif element_type == "hexa"
        mesh_faces = QuadFace[]
        for cell in grid.cells
            nodes = collect(cell.nodes)
            push!(mesh_faces, QuadFace(nodes[1], nodes[2], nodes[3], nodes[4])) 
            push!(mesh_faces, QuadFace(nodes[5], nodes[6], nodes[7], nodes[8])) 
            push!(mesh_faces, QuadFace(nodes[1], nodes[2], nodes[6], nodes[5]))  
            push!(mesh_faces, QuadFace(nodes[2], nodes[3], nodes[7], nodes[6]))  
            push!(mesh_faces, QuadFace(nodes[3], nodes[4], nodes[8], nodes[7]))  
            push!(mesh_faces, QuadFace(nodes[4], nodes[1], nodes[5], nodes[8]))  
        end
    end
end


# Visualization

# Define GLMakie view
figure = GLMakie.Figure(size = (1000,1000)) 
# ax = GLMakie.Axis(figure[1, 1], aspect = DataAspect()) 
ax = GLMakie.Axis3(figure[1, 1], aspect = :data)
xlims!(ax, -L/2, L/2)
ylims!(ax, -W/2, W/2)
zlims!(ax, -10, 10)

#Axis3 for z axis and figure[1,1] for first row and first column in the image layout
#reference: https://docs.makie.org/stable/reference/blocks/axis#Creating-an-Axis


vertices = Point3f[]

for n in grid.nodes
    x = n.x[1]
    y = n.x[2]
    if model_type == "2D"
        z = 0.0
    elseif model_type == "3D"
        z = n.x[3]
    end
    push!(vertices, Point3f(x, y, z))
end

if !isempty(mesh_faces)
    gb_mesh = GeometryBasics.Mesh(vertices, mesh_faces)
    GLMakie.mesh!(ax, gb_mesh, color = (:lightgreen, 1.0))
    GLMakie.wireframe!(ax, gb_mesh, color = :black, linewidth = 1.0)
end


GLMakie.display(figure)

cells = grid.cells
nodes = grid.nodes

# Quadrature rules
ip = Lagrange{RefQuadrilateral,1}()              # first order linear lagrange shape function
qr_inplane = QuadratureRule{RefQuadrilateral}(1) #(1 gauss point)
qr_ooplane = QuadratureRule{RefLine}(2)          # (2 gauss points for out of plane (thickness))
cv = CellValues(qr_inplane, ip, ip^3)            # evaluating shape functions and their derivatives at quadrature points
shape_reference_gradient(cv::CellValues, q_point, i) = cv.fun_values.dNdξ[i, q_point] #referencing shape functions to quadrature points


#region fibre coordinate system
function fiber_coordsys(Ps::Vector{Ferrite.Vec{3,Float64}})

    ef1 = Ferrite.Vec{3,Float64}[]
    ef2 = Ferrite.Vec{3,Float64}[]
    ef3 = Ferrite.Vec{3,Float64}[]
    for P in Ps
        a = abs.(P)
        j = 1
        if a[1] > a[3]; a[3] = a[1]; j = 2; end
        if a[2] > a[3]; j = 3; end

        e3 = P
        e2 = Tensors.cross(P, basevec(Ferrite.Vec{3}, j))
        e2 /= norm(e2)
        e1 = Tensors.cross(e2, P)

        push!(ef1, e1)
        push!(ef2, e2)
        push!(ef3, e3)
    end
    return ef1, ef2, ef3                        # Returns arrays of local axes for all nodes

end

#endregion

#region lamina coordinate system
function lamina_coordsys(dNdξ, ζ, x, p, h)

    e1 = zero(Ferrite.Vec{3})
    e2 = zero(Ferrite.Vec{3})

    for i in eachindex(dNdξ)
        e1 += dNdξ[i][1] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
        e2 += dNdξ[i][2] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
    end

    e1 /= norm(e1)
    e2 /= norm(e2)

    ez = Tensors.cross(e1,e2)
    ez /= norm(ez)

    a = 0.5*(e1 + e2)
    a /= norm(a)

    b = Tensors.cross(ez,a)
    b /= norm(b)

    ex = sqrt(2)/2 * (a - b)
    ey = sqrt(2)/2 * (a + b)

    return Tensor{2,3}(hcat(ex,ey,ez))          # returns a 3×3 tensor with columns (ex, ey, ez) → local lamina frame; Map strains and rotations into local lamina directions
end;

#endregion

#region getJacobian
function getjacobian(q, N, dNdξ, ζ, X, p, h)
    J = zeros(3,3)
    for a in eachindex(N)
        for i in 1:3, j in 1:3
            _dNdξ = (j==3) ? 0.0 : dNdξ[a][j]
            _dζdξ = (j==3) ? 1.0 : 0.0
            _N = N[a]

            J[i,j] += _dNdξ * X[a][i]  +  (_dNdξ*ζ + _N*_dζdξ) * h/2 * p[a][i]      # first term = in-plane mapping; second term = bending contribution
        end
    end

    return (q' * J) |> Tensor{2,3,Float64}              # rotates Jacobian into lamina coordinate system; Returns a 3×3 Jacobian tensor in local lamina frame
end

#endregion

#region strain matrix
function strain(dofvec::Vector{T}, N, dNdx, ζ, dζdx, q, ef1, ef2, h) where T        
    nnodes = length(N)
    u = [Ferrite.Vec{3,T}(dofvec[3*(i-1)+1:3*i]) for i in 1:nnodes]
    θ = [Ferrite.Vec{2,T}(dofvec[3*nnodes + 2*(i-1)+1 : 3*nnodes + 2*i]) for i in 1:nnodes]
    #u = reinterpret(Vec{3,T}, dofvec[1:3*nnodes])
    # θ = reinterpret(Vec{2,T}, dofvec[3*nnodes+1:5*nnodes])

    dudx = zeros(T, 3, 3)
    for m in 1:3, j in 1:3
        for a in eachindex(N)
            dudx[m,j] += dNdx[a][j] * u[a][m] + h/2 * (dNdx[a][j]*ζ + N[a]*dζdx[j]) * (θ[a][2]*ef1[a][m] - θ[a][1]*ef2[a][m])
        end
    end

    dudx = q*dudx
    ε = [
        dudx[1,1], 
        dudx[2,2], 
        dudx[1,2]+dudx[2,1], 
        dudx[2,3]+dudx[3,2], 
        dudx[1,3]+dudx[3,1]]   # ε[1] = ε_xx (membrane strain along x); ε[2] = ε_yy (membrane strain along y); ε[3]= γ_xy (in-plane shear); ε[4]= γ_yz (transverse shear in y-z plane); ε[5]= γ_xz (transverse shear in x-z plane)
    return ε
end

#endregion

#region stiffness integration
function integrate_shell!(ke, cv, qr_ooplane, X, data)
    nnodes = getnbasefunctions(cv)
    ndofs = nnodes*5
    h = data.thickness

    #Create the directors in each node.
    #Note: For a more general case, the directors should
    #be input parameters for the element routine.
    p = zeros(Ferrite.Vec{3}, nnodes)
    for i in 1:nnodes
        a = Ferrite.Vec{3}((0.0, 0.0, 1.0))
        p[i] = a/norm(a)
    end

    ef1, ef2, ef3 = fiber_coordsys(p) # defines a local coordinate system (ef1, ef2, ef3) for the shell (Important for rotation DOFs and bending terms)

    for iqp in 1:getnquadpoints(cv)
        N = [shape_value(cv, iqp, i) for i in 1:nnodes]                     # shape function values at quadrature point
        # dNdξ = [shape_reference_gradient(cv, iqp, i) for i in 1:nnodes]
        dNdξ = [shape_reference_gradient(cv, iqp, i) for i in 1:nnodes]     # shape function derivatives w.r.t reference coordinates ξ
        dNdx = [shape_gradient(cv, iqp, i) for i in 1:nnodes]               # shape function derivatives w.r.t physical coordinates x
                                                                            ## These are needed to compute strains and B-matrix
        for oqp in 1:length(qr_ooplane.weights)                             # Loop over quadrature points through thickness
            ζ = qr_ooplane.points[oqp][1]                                   # through-thickness location of quadrature point
            q = lamina_coordsys(dNdξ, ζ, X, p, h)                           # local lamina coordinate system at this thickness
                                                                            ## Important for Mindlin shell bending + shear contributions
            J = getjacobian(q, N, dNdξ, ζ, X, p, h)                             # Jacobian mapping reference → physical coordinates
            Jinv = inv(J)                                                       
            dζdx = Ferrite.Vec{3}((Jinv[3,1], Jinv[3,2], Jinv[3,3]))                    # derivative of through-thickness coordinate w.r.t physical coordinates
            # dNdx = [Ferrite.Vec{3,Float64}(Jinv * dNdξ[i]) for i in 1:nnodes]           # shape function derivatives w.r.t physical coordinates x
        
            #For simplicity, use automatic differentiation to construct the B-matrix from the strain.
            B = ForwardDiff.jacobian(
                (a) -> strain(a, N, dNdx, ζ, dζdx, q, ef1, ef2, h), zeros(Float64, ndofs) ) # automatically differentiates strain w.r.t nodal DOFs → gives B-matrix; computes strains from DOFs

            dV = qr_ooplane.weights[oqp] * getdetJdV(cv, iqp)               # quadrature weight × determinant of Jacobian (volume contribution)
        ke .+= B'*data.C*B * dV                                             # standard FEM stiffness contribution at this quadrature point
        end
    end
end

#endregion

#region material
κ = 5/6 # Shear correction factor
E = 210.0
ν = 0.3
a = (1-ν)/2
C = E/(1-ν^2) * [1 ν 0   0   0;
                ν 1 0   0   0;
                0 0 a*κ 0   0;
                0 0 0   a*κ 0;
                0 0 0   0   a*κ]


data = (thickness = t, C = C)

#endregion

#Define Dofs
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)



# ============================
# FEM (Plane Stress)
# ============================

# Material properties
E = 30e6  # psi
ν = 0.25

# Constitutive matrix for plane stress
D = (E / (1 - ν^2)) * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1-ν)/2]

println("D matrix:\n", D)

# Select first triangular element from mesh
elem_no = 1
cell = grid.cells[elem_no]
node_ids = collect(cell.nodes)
coords = [grid.nodes[id].x for id in node_ids]

x1, y1 = coords[1]
x2, y2 = coords[2]
x3, y3 = coords[3]

# x1, y1 = (0.0, -1.0)
# x2, y2 = (2.0, 0.0)
# x3, y3 = (0.0, 1.0)

# Calculate area
A = 0.5 * det([1 x1 y1; 1 x2 y2; 1 x3 y3])

# Build B-matrix
β1 = y2 - y3
β2 = y3 - y1
β3 = y1 - y2
γ1 = x3 - x2 
γ2 = x1 - x3
γ3 = x2 - x1

B = (1/(2*A)) * [β1   0.0  β2   0.0  β3   0.0; 0.0  γ1   0.0  γ2   0.0  γ3; γ1   β1   γ2   β2   γ3   β3]

# Element stiffness matrix
Ke = t * A * transpose(B) * D * B
println("Element stiffness matrix Ke:\n", Ke)

# Given nodal displacements (u1, v1, u2, v2, u3, v3)

# u_e = [0.0, 0.0025, 0.0012, 0.0, 0.0, 0.0025]  # inches


# Strains
ε = B * u_e
println("Strains:\n", ε)

# Stresses in (psi)
σ = D * ε
println("Stresses (σx, σy, τxy):\n", σ)


#Assembling in Global stiffness matrix

K = copy(Ke)  #considering only one element
F = zeros(6)

F[3] = -1000.0
F[4] = -1000.0

fixed_nodes = [1, 3]
fixed_dofs = Int[]
for n in fixed_nodes
    push!(fixed_dofs, 2*n - 1) 
    push!(fixed_dofs, 2*n)     
end
fixed_dofs = sort(unique(fixed_dofs))
all_dofs = collect(1:6)
free_dofs = setdiff(all_dofs, fixed_dofs)

Ke = K[free_dofs, free_dofs]
Fe = F[free_dofs]
d_free = Ke \ Fe

u_e = zeros(6)
u_e[free_dofs].= d_free

println("obtained nodal displacements u_e:")
println(" u1,v1 = ", u_e[1], ", ", u_e[2])
println(" u2,v2 = ", u_e[3], ", ", u_e[4])
println(" u3,v3 = ", u_e[5], ", ", u_e[6])
#inches

# Number of nodes in the mesh
num_nodes = length(grid.nodes)

# full displacement matrix (2 DOFs per node)
disp_u_e = zeros(2 * num_nodes)

# Obtained displacements into full matrix (only for 3 nodes only)
for i in 1:3
    disp_u_e[2*i - 1] = u_e[2*i - 1]
    disp_u_e[2*i] = u_e[2*i]
end

# Amplification factor for visualization
amplify = 1000000.0

# Create deformed vertices for ALL nodes
deformed_vertices = Point3f[]

for i in 1:num_nodes
    x = grid.nodes[i].x[1]
    y = grid.nodes[i].x[2]
    z = model_type == "2D" ? 0.0 : grid.nodes[i].x[3]

    u_disp = disp_u_e[2*i - 1]
    v_disp = disp_u_e[2*i]

    push!(deformed_vertices, Point3f(x + amplify * u_disp, y + amplify * v_disp, z))
end

# creating deformed mesh
deformed_mesh = GeometryBasics.Mesh(deformed_vertices, mesh_faces)

# Plot both original and deformed meshes

# Original mesh 
GLMakie.mesh!(ax, gb_mesh, color = (:lightgreen, 0.3))
GLMakie.wireframe!(ax, gb_mesh, color = :black, linewidth = 1.0)

# Deformed mesh 
GLMakie.mesh!(ax, deformed_mesh, color = (:lightgreen, 1.0))
GLMakie.wireframe!(ax, deformed_mesh, color = :red, linewidth = 1.5)

GLMakie.display(figure)