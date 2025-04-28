import vedo as v



def show_connect_points(mesh, idx, otheridx):
    mm = mesh.data['mesh']
    vert = mm.vertices
    face = mm.faces

    mesh = v.Mesh([vert.tolist(),face.tolist()],computeNormals=True)
    index = idx # pick one point
    pt = mesh.points(index)
    vtxs = mesh.points(otheridx)

    apt  = v.Point(pt, c="r", r=15)
    cpts = v.Points(vtxs, c="blue", r=15)

    v.show(mesh, apt, cpts, __doc__, bg='bb').close()

def show_select_points(mesh, idx):
    mm = mesh.data['mesh']
    vert = mm.vertices
    face = mm.faces
    mesh = v.Mesh([vert.tolist(),face.tolist()],computeNormals=True)
    index = idx # pick one point
    pt = mesh.points(index)
    apt  = v.Points(pt, c="r", r=15)
    v.show(mesh, apt, __doc__, bg='bb').close()

def trishow_select_points(mesh, idx):
    mm = mesh
    vert = mm.vertices
    face = mm.faces
    mesh = v.Mesh([vert.tolist(),face.tolist()],computeNormals=True)
    index = idx # pick one point
    if len(index) >0 :
        pt = mesh.points(index)
        apt  = v.Points(pt, c="r", r=15)
        v.show(mesh, apt, __doc__, bg='bb').close()
    else:
        print(idx,' is selected')

def show_mesh_scalar(mesh, scalar):
    assert mesh.data['mesh'].vertices.shape[0]==len(scalar)
    mesh = v.Mesh(mesh.data['mesh'])
    pts = mesh.points()
    points = v.Points(pts,r=10).cmap('rainbow',scalar)
    mesh.interpolateDataFrom(points, N=5).cmap('rainbow').addScalarBar()
    v.show(mesh,points, __doc__, axes=9).close()
    
def show_select_edge(mesh, select_edges):
    pass
