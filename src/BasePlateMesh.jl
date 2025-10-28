module BasePlateMesh

using Gmsh
using Ferrite
using FerriteGmsh

plate_dimensions = (L=300.0, W=300.0, t=10.0)
hole_dimensions = (hole_diameter=20.0, Sa1=0.0, Sa2=200.0) 



function create_base_plate_geometry(plate_dimensions, hole_dimensions)

    L, W = plate_dimensions
    hole_diameter, Sa1, Sa2 = hole_dimensions

    gmsh.initialize()
    gmsh.model.add("BasePlate")

    r = hole_diameter * 0.5

    # box_half = 1.0 * dia
    if Sa1 == 0.0
        hole_locations = [(-Sa2/2, Sa1/2), (Sa2/2, Sa1/2)]
    else
        hole_locations = [(-Sa2/2, -Sa1/2), (Sa2/2, -Sa1/2), (-Sa2/2, Sa1/2), (Sa2/2, Sa1/2)]
    end

    plate = gmsh.model.occ.addRectangle(-L/2, -W/2, 0.0, L, W)

    gmsh.model.occ.synchronize()

    hole_disks = [gmsh.model.occ.addDisk(x, y, 0.0, r, r) for (x, y) in hole_locations]

    gmsh.model.occ.synchronize()

    baseplate, _ = gmsh.model.occ.cut([(2, plate)],[(2, h) for h in hole_disks],-1,true,true)

    gmsh.model.occ.synchronize()

    return baseplate 

end



mesh_size = 5.0

function mesh_base_plate(mesh_size)
    size = mesh_size

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    gmsh.model.mesh.generate(2)
    gmsh.write("baseplate.msh")
    gmsh.finalize()

    grid = FerriteGmsh.togrid("baseplate.msh")

    mesh_faces = []
    for cell in grid.cells
        nodes = collect(cell.nodes)
        push!(mesh_faces, nodes)
    end

    return mesh_faces

end

end # module BasePlateMesh
