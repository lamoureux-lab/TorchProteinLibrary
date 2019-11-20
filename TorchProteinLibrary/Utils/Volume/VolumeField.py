import os
import sys
import torch
import enum

class FieldType(enum.Enum):
	scalar = 1
	vector = 3
	tensor = 5

class VolumeField():
	
	def __init__(self, T, resolution=1.0):
		self.T = T.transpose(1,2).transpose(2,3).transpose(1,2).contiguous()
		self.resolution = resolution
		self.ftype = None
		if self.T.dim() != 4:
			raise("Too many dimensions:", self.T.ndim())
		
		if self.T.size(0) == 1:
			self.ftype = FieldType.scalar
		elif self.T.size(0) == 3:
			self.ftype = FieldType.vector
		elif self.T.size(0) == 5:
			self.ftype = FieldType.tensor
		else:
			raise("Unknown field type:", self.T.size(0))

	def plot_scalar(self, contour_value=10.0):
		import vtk
		vol = vtk.vtkStructuredPoints()
		vol.SetDimensions(self.T.size(1),self.T.size(2),self.T.size(3))
		vol.SetOrigin(0,0,0)
		# vol.SetOrigin(self.T.size(1)*self.resolution/2.0, self.T.size(2)*self.resolution/2.0,self.T.size(3)*self.resolution/2.0)
		vol.SetSpacing(self.resolution, self.resolution, self.resolution)

		scalars = vtk.vtkDoubleArray()
		scalars.SetNumberOfComponents(1)
		scalars.SetNumberOfTuples(self.T.size(1)*self.T.size(2)*self.T.size(3))
		flat_tensor = self.T.view(self.T.size(1)*self.T.size(2)*self.T.size(3))
		for i in range(flat_tensor.size(0)):
			scalars.InsertTuple1(i, flat_tensor[i].item())
		vol.GetPointData().SetScalars(scalars)

		contour = vtk.vtkContourFilter()
		contour.SetInputData(vol)
		contour.SetValue(0, contour_value)

		volMapper = vtk.vtkPolyDataMapper()
		volMapper.SetInputConnection(contour.GetOutputPort())
		volMapper.ScalarVisibilityOff()
		
		actor = vtk.vtkActor()
		actor.SetMapper(volMapper)
		actor.GetProperty().EdgeVisibilityOn()
		actor.GetProperty().SetOpacity(0.3)
				
		return actor

	def plot_vector(self):
		import vtk
		vol = vtk.vtkStructuredPoints()
		vol.SetDimensions(self.T.size(1),self.T.size(2),self.T.size(3))
		vol.SetOrigin(0,0,0)
		# vol.SetSpacing(1.0/self.resolution, 1.0/self.resolution, 1.0/self.resolution)
		# vol.SetOrigin(self.T.size(1)*self.resolution/2.0, self.T.size(2)*self.resolution/2.0,self.T.size(3)*self.resolution/2.0)
		vol.SetSpacing(self.resolution, self.resolution, self.resolution)

		vectors = vtk.vtkDoubleArray()
		vectors.SetNumberOfComponents(3)
		vectors.SetNumberOfTuples(self.T.size(1)*self.T.size(2)*self.T.size(3))
		flat_tensor = self.T.view(3, self.T.size(1)*self.T.size(2)*self.T.size(3))
		for i in range(flat_tensor.size(1)):
			vectors.InsertTuple(i, [flat_tensor[0,i].item(), flat_tensor[1,i].item(), flat_tensor[2,i].item()] )
		vol.GetPointData().SetVectors(vectors)
		
		arrowSource = vtk.vtkArrowSource()

		glyph = vtk.vtkGlyph3D()
		glyph.SetInputData(vol)
		glyph.SetSourceConnection(arrowSource.GetOutputPort())

		glyph.SetVectorModeToUseVector()
		glyph.SetColorModeToColorByVector()
		glyph.SetScaleModeToScaleByVector()
		
		glyph.ScalingOn()
		glyph.OrientOn()
		
		glyph.Update()

		sgridMapper = vtk.vtkPolyDataMapper()
		sgridMapper.SetInputConnection(glyph.GetOutputPort())
		sgridActor = vtk.vtkActor()
		sgridActor.SetMapper(sgridMapper)


		return sgridActor

if __name__=='__main__':
	sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
	from Utils import VtkPlotter
	
	box_size = 30
	res = 1.0
	scalar_field = torch.zeros(1, 30, 30, 30)
	vector_field = torch.zeros(3, 30, 30, 30)
	for i in range(box_size):
		for j in range(box_size):
			for k in range(box_size):
				x = float(i)*res - float(box_size)*res/2.0
				y = float(j)*res - float(box_size)*res/2.0
				z = float(k)*res - float(box_size)*res/2.0
				scalar_field[0,i,j,k] = torch.sqrt(torch.tensor([x*x + y*y +z*z]))
				d2 = scalar_field[0,i,j,k]*scalar_field[0,i,j,k]
				vector_field[0,i,j,k] = torch.tensor([x])/(d2)
				vector_field[1,i,j,k] = torch.tensor([y])/(d2)
				vector_field[2,i,j,k] = torch.tensor([z])/(d2)
	scalar_field = VolumeField(scalar_field)
	vector_field = VolumeField(vector_field)
	

	# field.vtk_plot()
	plt = VtkPlotter()
	plt.add(scalar_field.plot_scalar())
	plt.add(vector_field.plot_vector())
	plt.show()