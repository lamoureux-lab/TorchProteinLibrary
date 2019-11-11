import vtk

class VtkPlotter:
	def __init__(self, window_size=(800, 640), window_name='VtkPlotter'):
		self.ren = vtk.vtkRenderer()
		self.renWin = vtk.vtkRenderWindow()
		self.renWin.AddRenderer(self.ren)
		self.iren = vtk.vtkRenderWindowInteractor()
		style = vtk.vtkInteractorStyleTrackballCamera()
		self.iren.SetInteractorStyle(style)
		self.iren.SetRenderWindow(self.renWin)
		self.renWin.SetSize(*window_size)
		self.renWin.SetWindowName(window_name)
		self.iren.Initialize()

		colors = vtk.vtkNamedColors()
		colors.SetColor("BkgColor", 0.9, 0.9, 0.9, 1.0)
		self.ren.SetBackground(colors.GetColor3d("BkgColor"))

	def show(self):
		self.ren.ResetCamera()
		self.ren.GetActiveCamera().Zoom(1.0)
		self.renWin.Render()
		self.iren.Start()

	def add(self, actor, color="Peacock"):
		colors = vtk.vtkNamedColors()
		actor.GetProperty().SetColor(colors.GetColor3d(color))
		self.ren.AddActor(actor)
