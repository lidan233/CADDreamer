
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


def get_camera_params(render):
    camera = render.GetActiveCamera()
    print(camera.GetPosition())
    print(camera.GetFocalPoint())
    print(camera.GetViewUp())

def render_mesh_scalar(mesh, scalar,min=None, max=None):
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
    bw_lut.SetTableRange(0,1)
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
            Colors.InsertNextTuple3(cormap[1][0],cormap[1][1],cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255,255)
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


def render_mesh_face_scalar(mesh, scalar,min=None, max=None ):
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
            Colors.InsertNextTuple3(cormap[1][0],cormap[1][1],cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255,255)
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



def render_edge_scalar(mesh, edges_scalar,min=None,max=None):
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
    bw_lut.SetTableRange(0,1)
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
        Colors.InsertNextTuple3(cormap[edgelabel][0],cormap[edgelabel][1],cormap[edgelabel][2])

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
        Colors.InsertNextTuple3(cormap[nodes[i]][0],cormap[nodes[i]][1],cormap[nodes[i]][2])

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



def render_face_color_normal(mm, facecolor):
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
            Colors.InsertNextTuple3(cormap[1][0],cormap[1][1],cormap[1][2])
        else:
            Colors.InsertNextTuple3(255, 255,255)
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
    def __init__(self,meshs,bounddir,showFace=True):
        colors = vtk.vtkNamedColors()
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.ren.SetBackground(colors.GetColor3d("Silver"))
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.bounds = [10000000.0,-1000000.0,10000000.0,-1000000.0,10000000.0,-1000000.0]
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
            if i==self.bounddir:
                center.append(bounds[2*self.bounddir]-THICKNESS*3)
                length.append(THICKNESS)
            else:
                center.append( (bounds[2*self.bounddir]+bounds[2*self.bounddir+1]) / 2.0 )
                length.append(bounds[2*self.bounddir+1] - bounds[2*self.bounddir] +rnge[i]*expand)

        #
        # plane = vtk.vtkCubeSource()
        # plane.SetCenter(center[0],center[1],center[2])
        #
        # plane.SetXLength(length[0])
        # plane.SetYLength(length[1])
        # plane.SetZLength(length[2])





        # planeMapper = vtk.vtkPolyDataMapper()
        # planeMapper.SetInputConnection(plane.GetOutputPort())
        # planeActor = vtk.vtkActor()
        # planeActor.SetMapper(planeMapper)

        #
        # self.ren.AddActor(planeActor)

        light1 = vtk.vtkLight()
        light1.SetFocalPoint((self.bounds[1] + self.bounds[0]) / 2.0,
                             (self.bounds[3] + self.bounds[2]) / 2.0,
                             (self.bounds[4] + self.bounds[5]) / 2.0)
        light1.SetPosition(self.bounds[0]*5,self.bounds[3],self.bounds[5]*5)
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



    def addMesh(self,mesh):
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
            if i % 2 ==0 and  bb[i]< self.bounds[i]:
                self.bounds[i] = bb[i]
            elif i % 2 ==1 and bb[i]>self.bounds[i]:
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
                Colors.InsertNextTuple3(t[0],t[1],t[2])
            trianglePolyData.GetPointData().SetScalars(Colors)

        if 'facecolor' in mesh.data.keys() and self.showFace:
            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName("Colors")
            assert mesh.data['facecolor'].shape[0] == mesh.data['faces'].shape[0]
            for i in range(len(mesh.data['faces'])):
                t = tuple(mesh.data['facecolor'][i])
                Colors.InsertNextTuple3(t[0],t[1],t[2])
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
        if 'facecolor' not in mesh.data.keys()  and 'pointcolor' not in mesh.data.keys():
            actor.GetProperty().SetColor(random.randint(0,255), random.randint(0,255), random.randint(0,255))

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


