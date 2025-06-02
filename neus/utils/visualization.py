import vedo as v
import vtk
import random
import numpy as np
import time
import os

random.seed(time.time())
segment_colors = np.array([
    [0, 114, 189],
    [217, 83, 26],
    [238, 177, 32],
    [126, 47, 142],
    [117, 142, 48],
    [76, 190, 238],
    [162, 19, 48],
    [240, 166, 202],
    [0, 255, 0]
])



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


def get_camera_params(render):
    camera = render.GetActiveCamera()
    print(camera.GetPosition())
    print(camera.GetFocalPoint())
    print(camera.GetViewUp())


def render_mesh_scalar(mesh, scalar, min=None, max=None):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    if min == None:
        min = np.min(scalar)
    if max == None:
        max = np.max(scalar)

    bw_lut = vtk.vtkLookupTable()
    bw_lut.SetTableRange(0, 1)
    bw_lut.SetSaturationRange(0, 0)
    bw_lut.SetHueRange(0, 0)
    bw_lut.SetValueRange(min, max)
    bw_lut.Build()

    temperature = vtk.vtkFloatArray()
    temperature.SetName("Temp")
    for t in scalar:
        temperature.InsertNextValue(t)

    trianglePolyData.GetPointData().SetScalars(temperature)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)
    mapper.SetLookupTable(bw_lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetLookupTable(actor.GetMapper().GetLookupTable())
    scalarBarActor.SetLabelFormat('%.2f')
    scalarBarActor.SetTitle("Temp")

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)
    ren.AddActor(scalarBarActor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_mesh_scalar_images(mesh, scalar, imagePath, min=None, max=None):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    if min == None:
        min = np.min(scalar)
    if max == None:
        max = np.max(scalar)

    bw_lut = vtk.vtkLookupTable()
    bw_lut.SetTableRange(0, 1)
    bw_lut.SetSaturationRange(0, 0)
    bw_lut.SetHueRange(0, 0)
    bw_lut.SetValueRange(min, max)
    bw_lut.Build()

    temperature = vtk.vtkFloatArray()
    temperature.SetName("Temp")
    for t in scalar:
        temperature.InsertNextValue(t)

    trianglePolyData.GetPointData().SetScalars(temperature)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)
    mapper.SetLookupTable(bw_lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetLookupTable(actor.GetMapper().GetLookupTable())
    scalarBarActor.SetLabelFormat('%.2f')
    scalarBarActor.SetTitle("Temp")

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)
    ren.AddActor(scalarBarActor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    screenshot(renWin, imagePath)
    get_camera_params(ren)


def screenshot(ren_win, filename):
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.Update()
    if filename is None:
        filename = 'screenshot'
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputData(w2if.GetOutput())
    writer.Write()


def render_simple_trimesh(mesh):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)
    #
    # actor1 = vtk.vtkActor()
    # actor1.SetMapper(mapper)
    # actor1.GetProperty().SetRepresentationToWireframe()
    # actor1.GetProperty().SetColor(255, 0, 0)
    # ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_simple_trimesh_select_nodes(mesh, nodes):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    cormap = []
    for i in range(1000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(mesh.vertices.shape[0]):
        if i in nodes:
            Colors.InsertNextTuple3(cormap[1][0], cormap[1][1], cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255, 255)
    trianglePolyData.GetPointData().SetScalars(Colors)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetPointSize(12);
    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_mesh_face_scalar(mesh, scalar, min=None, max=None):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    if min == None:
        min = np.min(scalar)
    if max == None:
        max = np.max(scalar)
    bw_lut = vtk.vtkLookupTable()
    bw_lut.SetTableRange(0, 1)
    bw_lut.SetSaturationRange(0, 0)
    bw_lut.SetHueRange(0, 0)
    bw_lut.SetValueRange(min, max)
    bw_lut.Build()

    temperature = vtk.vtkFloatArray()
    temperature.SetName("Temp")
    for t in scalar:
        temperature.InsertNextValue(t)

    trianglePolyData.GetCellData().SetScalars(temperature)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)
    mapper.SetScalarRange(trianglePolyData.GetScalarRange())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetLookupTable(actor.GetMapper().GetLookupTable())
    scalarBarActor.SetLabelFormat('%.2f')
    scalarBarActor.SetTitle("Temp")

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)
    ren.AddActor(scalarBarActor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_simple_trimesh_select_faces(mesh, faces):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    cormap = []
    for i in range(1000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(mesh.faces.shape[0]):
        if i in faces:
            Colors.InsertNextTuple3(cormap[1][0], cormap[1][1], cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255, 255)
    trianglePolyData.GetCellData().SetScalars(Colors)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_edge_scalar(mesh, edges_scalar, min=None, max=None):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p[0])
        line.GetPointIds().SetId(1, p[1])
        lines.InsertNextCell(line)

    linePolyData = vtk.vtkPolyData()
    linePolyData.SetPoints(points)
    linePolyData.SetLines(lines)

    if min == None:
        min = np.min(edges_scalar)
    if max == None:
        max = np.max(edges_scalar)

    bw_lut = vtk.vtkLookupTable()
    bw_lut.SetTableRange(0, 1)
    bw_lut.SetSaturationRange(0, 0)
    bw_lut.SetHueRange(0, 0)
    bw_lut.SetValueRange(min, max)
    bw_lut.Build()

    temperature = vtk.vtkFloatArray()
    temperature.SetName("Temp")
    for t in edges_scalar:
        temperature.InsertNextValue(t)

    linePolyData.GetCellData().SetScalars(temperature)

    bb = linePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(linePolyData)
    else:
        mapper.SetInputData(linePolyData)
    mapper.SetLookupTable(bw_lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_colorful_frame(mesh, edge_labels):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p[0])
        line.GetPointIds().SetId(1, p[1])
        lines.InsertNextCell(line)

    linePolyData = vtk.vtkPolyData()
    linePolyData.SetPoints(points)
    linePolyData.SetLines(lines)

    cormap = []
    for i in range(1000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(mesh.edges.shape[0]):
        edgelabel = int(edge_labels[i])
        Colors.InsertNextTuple3(cormap[edgelabel][0], cormap[edgelabel][1], cormap[edgelabel][2])

    linePolyData.GetCellData().SetScalars(Colors)

    bb = linePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(linePolyData)
    else:
        mapper.SetInputData(linePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_vertice_color(mesh, nodes):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    cormap = []
    for i in range(10000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(mesh.vertices.shape[0]):
        Colors.InsertNextTuple3(cormap[nodes[i]][0], cormap[nodes[i]][1], cormap[nodes[i]][2])

    trianglePolyData.GetPointData().SetScalars(Colors)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetPointSize(12);
    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_face_color(mm, facecolor):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mm.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mm.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    cormap = []
    for i in range(2000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)
    cormap = np.array(cormap)
    # cormap = segment_colors
    # cormap = np.array(cormap)
    face_color_index = [int(facecolor[i]) for i in range(mm.faces.shape[0])]
    facecolor = cormap[face_color_index]
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    assert facecolor.shape[0] == mm.faces.shape[0]
    for i in range(len(mm.faces)):
        t = tuple(facecolor[i])
        Colors.InsertNextTuple3(t[0], t[1], t[2])
    trianglePolyData.GetCellData().SetScalars(Colors)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)
    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    iren.Initialize()
    renWin.Render()
    iren.Start()
    get_camera_params(ren)


def render_simple_trimesh_image(mesh, dir, name):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1024, 1024)
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    ren.SetBackground(colors.GetColor3d("Silver"))
    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    renWin.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(os.path.join(dir, name))
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    get_camera_params(ren)


def render_face_color_image(mm, facecolor, dir, name):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1024, 1024)
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    ren.SetBackground(colors.GetColor3d("Silver"))
    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mm.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mm.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    # cormap = []
    # for i in range(1000):
    #     r = random.random() * 255
    #     b = random.random() * 255
    #     g = random.random() * 255
    #     color = (r, g, b)
    #     cormap.append(color)
    cormap = segment_colors
    cormap = np.array(cormap)
    face_color_index = [int(facecolor[i]) for i in range(mm.faces.shape[0])]
    facecolor = cormap[face_color_index]
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    assert facecolor.shape[0] == mm.faces.shape[0]
    for i in range(len(mm.faces)):
        t = tuple(facecolor[i])
        Colors.InsertNextTuple3(t[0], t[1], t[2])
    trianglePolyData.GetCellData().SetScalars(Colors)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)
    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    renWin.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(os.path.join(dir, name))
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    get_camera_params(ren)


def render_simple_to_meshcnn_image(mesh, dir, name):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ren.SetBackground(colors.GetColor3d("Black"))
    renWin.AddRenderer(ren)

    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    renWin.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(os.path.join(dir, name))
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    get_camera_params(ren)


def render_simple_trimesh_select_faces_image(mesh, faces, dir, name):
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1024, 1024)
    ren.SetBackground(colors.GetColor3d("Silver"))
    renWin.AddRenderer(ren)
    bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    for p in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, p[0])
        triangle.GetPointIds().SetId(1, p[1])
        triangle.GetPointIds().SetId(2, p[2])
        triangles.InsertNextCell(triangle)

    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    cormap = []
    for i in range(1000):
        r = random.random() * 255
        b = random.random() * 255
        g = random.random() * 255
        color = (r, g, b)
        cormap.append(color)

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(mesh.faces.shape[0]):
        if i in faces:
            Colors.InsertNextTuple3(cormap[1][0], cormap[1][1], cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255, 255)
    trianglePolyData.GetCellData().SetScalars(Colors)

    bb = trianglePolyData.GetBounds()
    for i in range(6):
        if i % 2 == 0 and bb[i] < bounds[i]:
            bounds[i] = bb[i]
        elif i % 2 == 1 and bb[i] > bounds[i]:
            bounds[i] = bb[i]

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetSpecular(0.51)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetAmbient(0.7)
    actor.GetProperty().SetSpecularPower(30.0)
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor1.GetProperty().SetRepresentationToWireframe()
    actor1.GetProperty().SetColor(255, 0, 0)
    ren.AddActor(actor1)

    light1 = vtk.vtkLight()
    light1.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light1.SetPosition(bounds[0] * 5, bounds[3], bounds[5] * 5)
    light1.SetIntensity(0.8)
    ren.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetFocalPoint((bounds[1] + bounds[0]) / 2.0,
                         (bounds[3] + bounds[2]) / 2.0,
                         (bounds[4] + bounds[5]) / 2.0)
    light2.SetPosition(bounds[1] * 10, bounds[3] * 10, bounds[5] * 10)
    light2.SetIntensity(0.9)
    ren.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    ren.SetPass(cameraP)
    renWin.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(os.path.join(dir, name))
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    get_camera_params(ren)


class Render:
    def __init__(self, meshs, bounddir, showFace=True):
        colors = vtk.vtkNamedColors()
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.ren.SetBackground(colors.GetColor3d("Silver"))
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.bounds = [10000000.0, -1000000.0, 10000000.0, -1000000.0, 10000000.0, -1000000.0]
        self.bounddir = bounddir
        self.mesh = meshs
        self.showFace = showFace

        for mesh in meshs:
            self.addMesh(mesh)

        bounds = self.bounds

        rnge = [0] * 3
        rnge[0] = bounds[1] - bounds[0]
        rnge[1] = bounds[3] - bounds[2]
        rnge[2] = bounds[5] - bounds[4]
        print("range: ", ', '.join(["{0:0.6f}".format(i) for i in rnge]))
        expand = 10.0
        THICKNESS = rnge[self.bounddir] * 0.1

        center = []
        length = []
        for i in range(3):
            if i == self.bounddir:
                center.append(bounds[2 * self.bounddir] - THICKNESS * 3)
                length.append(THICKNESS)
            else:
                center.append((bounds[2 * self.bounddir] + bounds[2 * self.bounddir + 1]) / 2.0)
                length.append(bounds[2 * self.bounddir + 1] - bounds[2 * self.bounddir] + rnge[i] * expand)


        light1 = vtk.vtkLight()
        light1.SetFocalPoint((self.bounds[1] + self.bounds[0]) / 2.0,
                             (self.bounds[3] + self.bounds[2]) / 2.0,
                             (self.bounds[4] + self.bounds[5]) / 2.0)
        light1.SetPosition(self.bounds[0] * 5, self.bounds[3], self.bounds[5] * 5)
        light1.SetIntensity(0.8)
        self.ren.AddLight(light1)

        light2 = vtk.vtkLight()
        light2.SetFocalPoint((self.bounds[1] + self.bounds[0]) / 2.0,
                             (self.bounds[3] + self.bounds[2]) / 2.0,
                             (self.bounds[4] + self.bounds[5]) / 2.0)
        light2.SetPosition(self.bounds[1] * 10, self.bounds[3] * 10, self.bounds[5] * 10)
        light2.SetIntensity(0.9)
        self.ren.AddLight(light2)

        shadows = vtk.vtkShadowMapPass()
        seq = vtk.vtkSequencePass()
        passes = vtk.vtkRenderPassCollection()
        passes.AddItem(shadows.GetShadowMapBakerPass())
        passes.AddItem(shadows)
        seq.SetPasses(passes)
        cameraP = vtk.vtkCameraPass()
        cameraP.SetDelegatePass(seq)

        self.ren.SetPass(cameraP)

    def render(self):
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def addMesh(self, mesh):
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()
        for p in mesh.data['coords']:
            points.InsertNextPoint(p[0], p[1], p[2])

        for p in mesh.data['faces']:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, p[0])
            triangle.GetPointIds().SetId(1, p[1])
            triangle.GetPointIds().SetId(2, p[2])
            triangles.InsertNextCell(triangle)

        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

        bb = trianglePolyData.GetBounds()
        for i in range(6):
            if i % 2 == 0 and bb[i] < self.bounds[i]:
                self.bounds[i] = bb[i]
            elif i % 2 == 1 and bb[i] > self.bounds[i]:
                self.bounds[i] = bb[i]

        cormap = []
        for i in range(1000):
            r = random.random() * 255
            b = random.random() * 255
            g = random.random() * 255
            color = (r, g, b)
            cormap.append(color)
        cormap = np.array(cormap)

        if 'pointcolor' not in mesh.data.keys() and 'pointlabel' in mesh.data.keys():
            point_color_index = [int(mesh.data['pointlabel'][i]) for i in range(mesh.data['mesh'].vertices.shape[0])]
            pointcolor = cormap[point_color_index]
            mesh.data['pointcolor'] = pointcolor

        if 'facecolor' not in mesh.data.keys() and 'facelabel' in mesh.data.keys():
            face_color_index = [int(mesh.data['facelabel'][i]) for i in range(mesh.data['mesh'].faces.shape[0])]
            facecolor = cormap[face_color_index]
            mesh.data['facecolor'] = facecolor

        if 'pointcolor' in mesh.data.keys() and not self.showFace:
            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName("Colors")
            assert mesh.data['pointcolor'].shape[0] == mesh.data['coords'].shape[0]
            for i in range(len(mesh.data['coords'])):
                t = tuple(mesh.data['pointcolor'][i])
                Colors.InsertNextTuple3(t[0], t[1], t[2])
            trianglePolyData.GetPointData().SetScalars(Colors)

        if 'facecolor' in mesh.data.keys() and self.showFace:
            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName("Colors")
            assert mesh.data['facecolor'].shape[0] == mesh.data['faces'].shape[0]
            for i in range(len(mesh.data['faces'])):
                t = tuple(mesh.data['facecolor'][i])
                Colors.InsertNextTuple3(t[0], t[1], t[2])
            trianglePolyData.GetCellData().SetScalars(Colors)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(trianglePolyData)
        else:
            mapper.SetInputData(trianglePolyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        color = np.random.randint(256, size=3)
        if 'facecolor' not in mesh.data.keys() and 'pointcolor' not in mesh.data.keys():
            actor.GetProperty().SetColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        actor.GetProperty().SetSpecular(0.51)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetAmbient(0.7)
        actor.GetProperty().SetSpecularPower(30.0)
        actor.GetProperty().SetOpacity(1.0)

        self.ren.AddActor(actor)

        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper)
        actor1.GetProperty().SetRepresentationToWireframe()
        actor1.GetProperty().SetColor(255, 0, 0)
        self.ren.AddActor(actor1)




import polyscope as ps
# import pymesh as pm
import trimesh as tri
import pyvista as pv

# def render_cadface_edges(cadfaces, edges):
import sys
# FREECADPATH = '/usr/local/lib'
# sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/lib'
sys.path.append(FREECADPATH)
FREECADPATH = '/usr/local/cuda-11.7/nsight-systems-2022.1.3/host-linux-x64'
sys.path.append(FREECADPATH)
import FreeCAD as App
import Part
import Mesh
from collections import deque
import torch
import trimesh.util
from typing import List
from pyvista import _vtk, PolyData
from numpy import split, ndarray
from neus.newton.FreeCADGeo2NewtonGeo import *
from neus.newton.newton_primitives import *
from neus.newton.process import  *

from tqdm import tqdm
from utils.util import *
from utils.visualization import *
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml
from scipy import stats

from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Display.SimpleGui import init_display
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.GeomProjLib import geomprojlib_Curve2d
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve


def render_seg_vertex_scalar(mesh, vertex_scalar):
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", vertex_scalar, defined_on='vertices', enabled=True)
    ps.show()
def render_seg_face_scalar(mesh, face_scalar):
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", face_scalar, defined_on='faces', enabled=True)
    ps.show()

def render_seg_select_face(mesh, select_face_idx):
    face_scalar = np.zeros(len(mesh.faces))
    face_scalar[select_face_idx] = 1
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", face_scalar, defined_on='faces', enabled=True)
    ps.show()


def render_seg_select_vertices(mesh, select_vertex_idx):
    vertex_scalar = np.zeros(len(mesh.vertices))
    vertex_scalar[select_vertex_idx] = 1

    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_scalar_quantity("my scalar", vertex_scalar, defined_on='vertices', enabled=True)
    ps.show()


def render_all_patches(mesh, label_components):
    ps.init()
    ps.remove_all_structures()
    count = 0
    for label in label_components:
        patch_mesh = mesh.submesh([label])[0]
        ps.register_surface_mesh("mesh"+ str(count), patch_mesh.vertices, patch_mesh.faces, smooth_shade=True)
        count += 1

    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()


from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
def get_edge_endpoints(edge):
    start_vertex = gp_Pnt()
    end_vertex = gp_Pnt()
    curve_adaptor = BRepAdaptor_Curve(edge)
    curve_adaptor.D0(curve_adaptor.FirstParameter(), start_vertex)
    curve_adaptor.D0(curve_adaptor.LastParameter(), end_vertex)
    return start_vertex, end_vertex


def occV2arr(current_v):
    current_point = BRep_Tool.Pnt(current_v)
    p_arr = np.array([current_point.X(), current_point.Y(), current_point.Z()])
    return p_arr



# 辅助函数：计算两个点之间的欧几里德距离
def distance_between_points(point1, point2):
    return point1.Distance(point2)


# 主函数：计算两个边缘之间的距离
def distance_between_edges(edge1, edge2):
    start_vertex1, end_vertex1 = get_edge_endpoints(edge1)
    start_vertex2, end_vertex2 = get_edge_endpoints(edge2)
    distance_start_to_start = distance_between_points(start_vertex1, start_vertex2)
    distance_start_to_end = distance_between_points(start_vertex1, end_vertex2)
    distance_end_to_start = distance_between_points(end_vertex1, start_vertex2)
    distance_end_to_end = distance_between_points(end_vertex1, end_vertex2)
    return min(distance_start_to_start, distance_start_to_end) + min(distance_end_to_start, distance_end_to_end)


def discretize_edge_distance(edge1, edge2, num_points=5):
    edge_shape = topods.Edge(edge1)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)
    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()
    step = (u_end - u_start) / num_points
    discretized_points = []
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append([point.X(), point.Y(), point.Z()])
    discretized_points1 = np.array(discretized_points)

    edge_shape = topods.Edge(edge2)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)
    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()
    step = (u_end - u_start) / num_points
    discretized_points = []
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append([point.X(), point.Y(), point.Z()])
    discretized_points2 = np.array(discretized_points)

    dis1 = np.abs(discretized_points1 - discretized_points2).mean()
    dis2 = np.abs(discretized_points1 - discretized_points2[::-1]).mean()
    print(dis1, dis2)
    return min(dis1, dis2)

def discretize_edge(edge, num_points=100):
    edge_shape = topods.Edge(edge)
    adaptor_curve = BRepAdaptor_Curve(edge_shape)

    # 获取曲线的参数范围
    u_start = adaptor_curve.FirstParameter()
    u_end = adaptor_curve.LastParameter()

    # 计算步长
    step = (u_end - u_start) / num_points

    # 存储离散化后的点
    discretized_points = []

    # 在参数范围内获取离散化点
    for i in range(num_points + 1):
        u = u_start + i * step
        point = adaptor_curve.Value(u)
        discretized_points.append(gp_Pnt(point.X(), point.Y(), point.Z()))

    return discretized_points

def calculate_edge_length(edges):
    total_length = 0.0
    for edge in edges:
        curve_adaptor = BRepAdaptor_Curve(edge)
        length = GCPnts_AbscissaPoint().Length(curve_adaptor)
        total_length += length
    return total_length



def discretize_edges(edges, total_points):
    edge_lengths = [calculate_edge_length([ee]) for ee in edges]
    total_length = np.sum(edge_lengths)

    discretized_wire = []
    remaining_points = total_points

    for i, edge in enumerate(edges):
        if i == len(edges) - 1:
            edge_points = remaining_points
        else:
            edge_ratio = edge_lengths[i] / total_length
            edge_points = max(2, int(round(total_points * edge_ratio)))
            remaining_points -= edge_points

        edge_discretized = discretize_edge(edge, edge_points)
        if i > 0:
            edge_discretized = edge_discretized[1:]
        discretized_wire.extend(edge_discretized)
    return discretized_wire

def face_to_trimesh(face, linear_deflection=0.001):

    bt = BRep_Tool()
    BRepMesh_IncrementalMesh(face, linear_deflection, True)
    location = TopLoc_Location()
    facing = bt.Triangulation(face, location)
    if facing is None:
        return None
    triangles = facing.Triangles()

    vertices = []
    faces = []
    offset = face.Location().Transformation().Transforms()

    for i in range(1, facing.NbNodes() + 1):
        node = facing.Node(i)
        coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
        # coord = [node.X(), node.Y() , node.Z() ]
        vertices.append(coord)

    for i in range(1, facing.NbTriangles() + 1):
        triangle = triangles.Value(i)
        index1, index2, index3 = triangle.Get()
        tface = [index1 - 1, index2 - 1, index3 - 1]
        faces.append(tface)
    tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)


    return tmesh

def getVertex(compound):
    vs = []
    explorer = TopExp_Explorer(compound, TopAbs_VERTEX)
    while explorer.More():
        current_v = topods.Vertex(explorer.Current())
        vs.append(current_v)
        explorer.Next()
    return vs


def getWires(compound):
    wires = []
    wire_explorer = TopExp_Explorer(compound, TopAbs_WIRE)
    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        wires.append(wire)
        wire_explorer.Next()

    return wires

def getEdges(compound):
    edges = []
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        current_edge = topods.Edge(explorer.Current())
        edges.append(current_edge)
        explorer.Next()
    return edges

def getFaces(compound):
    faces = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        current_face = topods.Face(explorer.Current())
        faces.append(current_face)
        explorer.Next()
    return faces

def face2wire(face):
    c_wire = BRepBuilderAPI_MakeWire()


    for edge in getEdges(face):
        e = edge.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)
    wire = c_wire.Wire()
    return wire


def render_all_occ(cad_faces=None, cad_edges=None, cad_vertices=None, select_edge_idx=None):
    mesh_face_label = None
    meshes = None
    if cad_faces is not None:
        meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
        mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]
    output_edges = None
    if cad_edges is not None:
        real_edges = []
        for ce in cad_edges:
            real_edges += getEdges(ce)
        discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
        output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
    output_vertices = None
    if output_vertices is not None:
        output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
    render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices, meshes_label=mesh_face_label)


def Compound(faces):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)

    for face in faces:
        explorer = TopExp_Explorer(face, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            builder.Add(compound, face)
            explorer.Next()

    return compound


def render_occ_faces_and_primitives(cad_faces=None, cad_edges=None, cad_vertices=None, primitives = None, select_edge_idx=None):
    mesh_face_label = None
    meshes = None
    primitive_meshes = None
    if cad_faces is not None:
        meshes = [face_to_trimesh(ccf) for cf in cad_faces for ccf in getFaces(cf)]
        mesh_face_label = [np.ones(len(meshes[i].faces)) * i for i in range(len(meshes))]

    if primitives is not None:
        primitive_meshes = [face_to_trimesh(cf) for cf in primitives ]


    output_edges = None
    if cad_edges is not None:
        real_edges = []
        for ce in cad_edges:
            real_edges += getEdges(ce)
        discrete_edges = [discretize_edge(ce) if ce.Orientation() != TopAbs_REVERSED else discretize_edge(ce)[::-1] for ce in real_edges ]
        output_edges = [np.array([list(p.Coord()) for p in edge]) for edge in discrete_edges]
    output_vertices = None
    if cad_vertices is not None:
        output_vertices = np.array([occV2arr(current_v) for current_v in cad_vertices ])
    render_mesh_path_points(meshes=meshes, edges=output_edges, points=output_vertices,
                            meshes_label=mesh_face_label, primitive_meshes=primitive_meshes)


def render_mesh_path_points(meshes=None, edges=None, points=None, select_vertex=None,
                            select_face=None, select_edges=None, select_points=None,
                            meshes_label=None, primitive_meshes=None):
    ps.init()
    ps.remove_all_structures()
    radius = 0.003
    if meshes is not None:
        final_mesh = tri.util.concatenate(meshes)
        all_mesh = ps.register_surface_mesh("mesh", final_mesh.vertices, final_mesh.faces, smooth_shade=True)
        if meshes_label is not None:
            all_mesh.add_scalar_quantity("cad_face_label_scalar", np.concatenate(meshes_label), defined_on='faces')

    if primitive_meshes is not None:
        for i in range(len(primitive_meshes)):
            primitive_mesh = primitive_meshes[i]
            pmesh = ps.register_surface_mesh("primitive"+str(i), primitive_mesh.vertices,
                                             primitive_mesh.faces, transparency=0.3,
                                             smooth_shade=True)


    if edges is not None :
        discrete_edges_numpy = edges
        all_edge_meshes = []
        all_edge_meshes_values = []
        for tpoints in discrete_edges_numpy:
            polydata = pv.PolyData(tpoints)
            p_values = np.arange(len(tpoints))
            polydata.lines = np.hstack(([len(tpoints)], np.arange(len(tpoints))))
            tube = polydata.tube(radius=0.01, n_sides=18)
            tube = tube.triangulate()
            vpoints = tube.points
            faces = tube.faces.reshape(-1, 4)[:,1:]

            from scipy.spatial import KDTree
            tree = KDTree(tpoints)
            distance, index = tree.query(vpoints)
            v_value = p_values[index]
            edge_mesh = tri.Trimesh(vpoints, faces, process=False)
            all_edge_meshes.append(edge_mesh)
            all_edge_meshes_values.append(v_value)

        final_edge_mesh = tri.util.concatenate(all_edge_meshes)
        edge_mesh = ps.register_surface_mesh("edges", final_edge_mesh.vertices, final_edge_mesh.faces, smooth_shade=True)
        edge_mesh.add_scalar_quantity("vertex_scalar", np.concatenate(all_edge_meshes_values), defined_on='vertices')

    if points is not None:
        final_vertex = points
        ps.register_point_cloud("points", final_vertex, radius=radius * 1.2)
    elif edges is not None:
        discrete_edges_numpy = edges
        final_vertices = []
        for tpoints in discrete_edges_numpy:
            final_vertices.append(tpoints[0])
            final_vertices.append(tpoints[-1])
        ps.register_point_cloud("points", np.array(final_vertices), radius=radius * 1.2)

    if select_face is not None:
        mesh_label = np.zeros(len(final_mesh.faces))
        mesh_label[select_face] = 1
        ps.get_surface_mesh("mesh").add_scalar_quantity("flabel",
                                                                np.array(mesh_label),
                                                                defined_on='faces',
                                                                cmap='turbo', enabled=True)
    if select_vertex is not None:
        mesh_vlabel = np.zeros(len(final_mesh.vertices))
        mesh_vlabel[select_vertex] = 1
        ps.get_surface_mesh("mesh").add_scalar_quantity("fvlabel",
                                                                np.array(mesh_vlabel),
                                                                defined_on='vertices',
                                                                cmap='turbo', enabled=True)

    if select_edges is not None:
        selected_edges_data = [edges[select_edges[i]] for i in range(len(select_edges))]
        selected_out_edges = []
        select_radius = radius * 1.1
        discrete_edges_numpy = selected_edges_data
        for points in discrete_edges_numpy:
            polydata = pv.PolyData(points)
            polydata.lines = np.hstack(([len(points)], np.arange(len(points))))
            tube = polydata.tube(radius=select_radius, n_sides=18)
            tube = tube.triangulate()
            points = tube.points
            faces = tube.faces.reshape(-1, 4)[:, 1:]
            edge_mesh = tri.Trimesh(points, faces, process=False)
            selected_out_edges.append(edge_mesh)
        final_select_edges = tri.util.concatenate(selected_out_edges)
        ps.register_surface_mesh("select_edges", final_select_edges.vertices, final_select_edges.faces, smooth_shade=True)

    if select_points is not None:
        ps.get_point_cloud("points").add_scalar_quantity("selectv",
                                                                np.array(select_points),
                                                                cmap="turbo", enabled=True)

    ps.set_shadow_darkness(0.2)
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode('shadow_only')
    ps.show()



