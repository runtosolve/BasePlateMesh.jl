# using Gmsh

using Gmsh: Gmsh, gmsh

Gmsh.initialize()
gmsh.model.add("BasePlate")

# Plate dimensions
L, W, t = 300.0, 300.0, 10.0

# hole radius
dia = 20.0
r = dia*0.5
element_type = "tri"   # "quad" for quadrilateral or "tri" for triangular, "tetra" for tetrahedron, "hexa" for hexahedron
model_type = "3D"       # "2D" or "3D"

plate = gmsh.model.occ.addRectangle(-L/2, -W/2, 0.0, L, W)


    plate = gmsh.model.occ.addRectangle(-L/2, -W/2, 0.0, L, W)



box_half = 1.0 * dia
hole_locations = [(-100.0, -100.0), (100.0, -100.0), (-100.0, 100.0), (100.0, 100.0)]
