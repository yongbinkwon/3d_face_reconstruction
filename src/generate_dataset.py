from Dataset.dataset_generator.generate_dataset import generate_facegen_dataset, generate_rest, fix_dataset, generate_rotated, sort_dataset
import os

if __name__=='__main__':
    #generate_facegen_dataset("/lhome/yongbk/facegen/subjects", "/lhome/yongbk/dataset/facegen", "/lhome/yongbk/dataset/file_list.txt")
    sort_dataset("/lhome/yongbk/facegen/subjects", "/lhome/yongbk/dataset/facegen", "/lhome/yongbk/dataset/file_list_sorted.txt")
    #generate_facegen_dataset(os.path.expanduser('~/testdataset'), "/mnt/lhome/lhome/yongbk/dataset/facegen", "/mnt/lhome/lhome/yongbk/dataset/facegen/file_list.txt")
    #fix_dataset("/mnt/lhome/lhome/yongbk/facegen/subjects", "/mnt/lhome/lhome/yongbk/dataset/facegen", "/mnt/lhome/lhome/yongbk/dataset/facegen/file_list.txt")
    #generate_rotated("/disk2/lhome/lhome/yongbk/dataset/facegen/file_list.txt")