import scipy.io as sio
import csv
import glob
import os


def matlab2csv(matlab_file_path, csv_file_path):
    ''' Search for all the .mat file in the given matlab_file_path, and coverts
    them to the CSV file.
    :param matlab_file_path: Directory containting the .mat files
    :param csv_file_path: Directory to save the converted files 
    '''
    files = glob.glob(os.path.join(matlab_file_path, "*.mat"))
    for file in files:
        id = os.path.splitext(os.path.basename(file))[0]
        csv_filename = "%d.csv" % int(id)
        print(id)
        content = sio.loadmat(file)
        out = os.path.join(csv_file_path, csv_filename)
        with open(out,"w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(content['lines'])
