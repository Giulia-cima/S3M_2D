import  os


path_origin = "/home/idrologia/share/FloodProofs_v3/S3M/S3M_HS_ground/archivio/"
# list all the folder in the path
list_years = os.listdir(path_origin)
# sort the list
list_years.sort()

for year in list_years:
    # list all the folder in the year folder
    list_months = os.listdir(path_origin + year)
    # sort the list
    list_months.sort()
    for month in list_months:
        # list all the folder in the month folder
        list_days = os.listdir(path_origin + year + "/" + month)
        # sort the list
        list_days.sort()
        for day in list_days:
            # list all the files in the day folder
            list_files = os.listdir(path_origin + year + "/" + month + "/" + day)
            # sort the list
            list_files.sort()
            for file in list_files:
                if file.endswith(".tif"):
                    print(year + "/" + month + "/" + day + "/" + file)