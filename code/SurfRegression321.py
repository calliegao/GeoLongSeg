#!/usr/bin/env python
import numpy as np
import os
import nibabel as nb
import argparse
import shutil
from pathlib import Path
import vtk
import xml.etree.ElementTree as ET



def get_parser():
    parser = argparse.ArgumentParser(description='Geodesic Regression')

    parser.add_argument('--tp0', type=str, required=True, help='tp0 name')
    parser.add_argument('--tp1', type=str, required=True, help='tp1 name')
    parser.add_argument('--tp2', type=str, required=True, help='tp2 name')
    parser.add_argument('--cu', type=str, default='./output1', help='cu out dir')
    args = parser.parse_args()

    return args


def change_one_xml(xml_path, xml_dw, update_content):
    doc = ET.parse(xml_path)
    root = doc.getroot()
    sub1 = root.find(xml_dw)
    sub1.text = update_content
    doc.write(xml_path)


def CalculateSurfDist(RegressedSurf, OriginalSurf):
    surf_reader = vtk.vtkPolyDataReader()
    surf_reader.SetFileName(RegressedSurf)
    surf_reader.Update()
    RegressedSurf_vtk = surf_reader.GetOutput()

    surf_reader = vtk.vtkPolyDataReader()
    surf_reader.SetFileName(OriginalSurf)
    surf_reader.Update()
    OriginalSurf_vtk = surf_reader.GetOutput()

    num_pt = RegressedSurf_vtk.GetNumberOfPoints()
    pt_dist = 0
    for i in range(num_pt):
        pt = np.array(RegressedSurf_vtk.GetPoint(i))
        p_closestPoint = np.array(OriginalSurf_vtk.GetPoint(i))
        pt_dist = pt_dist + np.linalg.norm(pt - p_closestPoint)

    return pt_dist


def nii_2_mesh(filename_nii, vtk_filename, stl_filename, label):
    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(30)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(stl_filename)
    writer.Write()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_filename)
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.Update()


def GeodesicError(workpath, label, start_kw):
    RegressionOutputdir = workpath + "/RegressionOutput"

    ## Generate surfaces for the labels
    niiInputdir = workpath + "/nii"
    Surfdir = workpath + "/surf"
    Surffile = Path(Surfdir)

    for case_folder in os.listdir(niiInputdir):
        case_dir = os.path.join(niiInputdir, case_folder)
        newfilename = case_folder[0:3]
        Surfname_vtk = Surfdir + "/" + newfilename + ".vtk"
        Surfname_stl = Surfdir + "/" + newfilename + ".stl"
        nii_2_mesh(case_dir, Surfname_vtk, Surfname_stl, label)

    data_xml = workpath + "/data_set.xml"
    xml_dw = './/subject[@id="sub_test"]/visit[@id="hippo_t0"]/filename[@object_id="hippo"]'
    update_content = workpath + "/surf/tp0.vtk"
    change_one_xml(data_xml, xml_dw, update_content)

    xml_dw = './/subject[@id="sub_test"]/visit[@id="hippo_t1"]/filename[@object_id="hippo"]'
    update_content = workpath + "/surf/tp1.vtk"
    change_one_xml(data_xml, xml_dw, update_content)

    xml_dw = './/subject[@id="sub_test"]/visit[@id="hippo_t2"]/filename[@object_id="hippo"]'
    update_content = workpath + "/surf/tp2.vtk"
    change_one_xml(data_xml, xml_dw, update_content)

    model_xml = workpath + "/model.xml"
    xml_dw = './/template/object[@id="hippo"]/filename'
    update_content = workpath + "/surf/tp0.vtk"
    change_one_xml(model_xml, xml_dw, update_content)

    print('Geodesic regression start...')

    cmd6 = "deformetrica estimate " + model_xml + " " + data_xml + " -p " + workpath + "/optimization_parameters.xml -o " + RegressionOutputdir

    dist = 0
    N = 10
    eps = 1

    for cyclnum in range(N):

        if dist < eps:

            print('Regression round: %s' % str(cyclnum + 1))

            xml_dw = './/deformation-parameters/kernel-width'
            new_kw = start_kw + cyclnum * 0.5
            update_content = str(new_kw)
            change_one_xml(model_xml, xml_dw, update_content)

            os.system(cmd6)

            # Compute regression results. If unscessed, try to repair model parameter kernel-width
            RegressedSurf1dir = RegressionOutputdir + "/GeodesicRegression__GeodesicFlow__hippo__tp_0__age_0.00.vtk"
            Originaltp0dir = workpath + "/surf/tp0.vtk"
            RegressedSurf2dir = RegressionOutputdir + "/GeodesicRegression__GeodesicFlow__hippo__tp_1__age_1.00.vtk"
            Originaltp1dir = workpath + "/surf/tp1.vtk"
            RegressedSurf3dir = RegressionOutputdir + "/GeodesicRegression__GeodesicFlow__hippo__tp_2__age_2.00.vtk"
            Originaltp2dir = workpath + "/surf/tp2.vtk"

            surf_reader = vtk.vtkPolyDataReader()
            surf_reader.SetFileName(Originaltp0dir)
            surf_reader.Update()
            Originaltp0dir_vtk = surf_reader.GetOutput()
            num_pt0 = Originaltp0dir_vtk.GetNumberOfPoints()

            surf_reader = vtk.vtkPolyDataReader()
            surf_reader.SetFileName(Originaltp1dir)
            surf_reader.Update()
            Originaltp1dir_vtk = surf_reader.GetOutput()
            num_pt1 = Originaltp1dir_vtk.GetNumberOfPoints()

            surf_reader = vtk.vtkPolyDataReader()
            surf_reader.SetFileName(Originaltp2dir)
            surf_reader.Update()
            Originaltp2dir_vtk = surf_reader.GetOutput()
            num_pt2 = Originaltp2dir_vtk.GetNumberOfPoints()

            dist1 = CalculateSurfDist(RegressedSurf1dir, Originaltp0dir) / num_pt0
            dist2 = CalculateSurfDist(RegressedSurf2dir, Originaltp1dir) / num_pt1
            dist3 = CalculateSurfDist(RegressedSurf3dir, Originaltp2dir) / num_pt2
            TotalDist = (dist1 + dist2 + dist3) / np.mean(num_pt0 + num_pt1 + num_pt2)
            dist = CalculateSurfDist(RegressedSurf3dir, RegressedSurf1dir)
            print('Geodesic error is: %s' % str(dist))
        else:
            print('Regression Succeed!')
            print('Geodesic loss of Surf(tp0) is: %s' % str(dist1))
            print('Geodesic loss of Surf(tp1) is: %s' % str(dist2))
            print('Geodesic loss of Surf(tp2) is: %s' % str(dist3))
            print('Geodesic loss is: %s' % str(TotalDist))
            break
    return TotalDist, new_kw

def vtk2niisurf(rgo_vtk, old_nii ,out_path):
    img = nb.load(old_nii).get_fdata()
    affine = nb.load(old_nii).affine
    hdr = nb.load(old_nii).header
    name = os.path.basename(rgo_vtk)
    name = name.split("_sf.")[0]
    print(name)
    pt_reader = vtk.vtkPolyDataReader()
    pt_reader.SetFileName(rgo_vtk)
    pt_reader.Update()
    pt_vtk = pt_reader.GetOutput()
    num_pt = pt_vtk.GetNumberOfPoints()

    pt_space = np.zeros(img.shape)

    for i in range(num_pt):
        pt = np.array(pt_vtk.GetPoint(i))
        pt_space[int(pt[0]), int(pt[1]), int(pt[2])] = 1

    pt_space_int8 = pt_space.astype(np.uint8)
    img_nii = nb.Nifti1Image(pt_space_int8, affine, hdr)
    nb.save(img_nii, out_path + '/' + str(name) + '_surf.nii.gz')


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    workpath = '/data03/cr7/pycharm/pythonProject230221/dfca'
    label = 1
    start_kw = 7

    aim = get_parser()
    cu_outdir = aim.cu
    nii_dir = workpath + '/nii'
    rgo_dir = workpath + '/RegressionOutput'  # RegressionOutputdir
    rgsurf_vtk_dir = workpath + '/rgsurf_out_vtk'  # regression output surf nii
    rgsurf_nii_dir = workpath + '/rgsurf_out_nii'

    sub_tp0 = aim.tp0
    path_old_tp0 = cu_outdir + '/' + sub_tp0 + '_cu.nii.gz'
    path_new_tp0 = nii_dir + '/' + 'tp0.nii.gz'
    shutil.copy(path_old_tp0, path_new_tp0)

    sub_tp1 = aim.tp1
    path_old_tp1 = cu_outdir + '/' + sub_tp1 + '_cu.nii.gz'
    path_new_tp1 = nii_dir + '/' + 'tp1.nii.gz'
    shutil.copy(path_old_tp1, path_new_tp1)

    sub_tp2 = aim.tp2
    path_old_tp2 = cu_outdir + '/' + sub_tp2 + '_cu.nii.gz'
    path_new_tp2 = nii_dir + '/' + 'tp2.nii.gz'
    shutil.copy(path_old_tp2, path_new_tp2)

    TotalDist, new_kw = GeodesicError(workpath, label, start_kw)

    rgo_tp0 = '/GeodesicRegression__Reconstruction__hippo__tp_0__age_0.00.vtk'
    rgo_tp1 = '/GeodesicRegression__Reconstruction__hippo__tp_1__age_1.00.vtk'
    rgo_tp2 = '/GeodesicRegression__Reconstruction__hippo__tp_2__age_2.00.vtk'
    rgo_tp0_newpath = rgsurf_vtk_dir + '/' + sub_tp0 + '_sf.vtk'
    rgo_tp1_newpath = rgsurf_vtk_dir + '/' + sub_tp1 + '_sf.vtk'
    rgo_tp2_newpath = rgsurf_vtk_dir + '/' + sub_tp2 + '_sf.vtk'
    shutil.copy(rgo_dir + rgo_tp0, rgo_tp0_newpath)
    shutil.copy(rgo_dir + rgo_tp1, rgo_tp1_newpath)
    shutil.copy(rgo_dir + rgo_tp2, rgo_tp2_newpath)

    vtk2niisurf(rgo_tp0_newpath, path_new_tp0, rgsurf_nii_dir)
    vtk2niisurf(rgo_tp1_newpath, path_new_tp1, rgsurf_nii_dir)
    vtk2niisurf(rgo_tp2_newpath, path_new_tp2, rgsurf_nii_dir)



