import vtk

def nii_2_mesh (filename_nii, Output_filename, label):

    """
    Read a nifti file including a binary map of a segmented organ with label id = label. 
    Convert it to a smoothed mesh of type stl.

    filename_nii     : Input nifti binary map 
    filename_stl     : Output mesh name in stl format
    label            : segmented label id 
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    
    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()
    
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(30) 
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()
     
    # save the output
    #writer = vtk.vtkSTLWriter()
    #writer.SetInputConnection(smoother.GetOutputPort())
    #writer.SetFileTypeToASCII()
    #writer.SetFileName(Output_filename)
    #writer.Write()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(Output_filename)
    writer.SetInputConnection(smoother.GetOutputPort())
    #writer.SetInputData(smoother.GetOutputPort())
    #writer.SetDataModeToAscii()
    writer.Update()

if __name__ == '__main__':
    
    filename_nii =  '/home/hitlab/deformetrica/examples/regression/landmark/3d/LongSeg/nii/n69tp0hpLb.nii.gz'
    Output_filename = '/home/hitlab/deformetrica/examples/regression/landmark/3d/LongSeg/nii/tp0.vtk'
    label = 75
    nii_2_mesh (filename_nii, Output_filename, label)
