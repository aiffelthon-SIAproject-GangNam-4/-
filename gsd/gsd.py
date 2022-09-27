from osgeo import gdal

fileformat = "GTiff"
# data_path = "./gsd-50cm.tif"
data_path = "./4561.tif"
raster = gdal.Open(data_path)

gt = raster.GetGeoTransform()
print("pixel size for X: \n", gt[1])
print("pixel size for Y: \n", -gt[5])