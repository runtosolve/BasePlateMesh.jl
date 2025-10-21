# Preamble:
using BasePlateMesh
using Test

@testset "base plate geometry" begin 
   
    plate_dimensions = (L=300.0, W=300.0, t=10.0)
    hole_dimensions = (hole_diameter=20.0, Sa1=0.0, Sa2=200.0) 

    baseplate = BasePlateMesh.create_base_plate_geometry(plate_dimensions, hole_dimensions)

    @test baseplate == 9
    #TODO - write boolean here 

end