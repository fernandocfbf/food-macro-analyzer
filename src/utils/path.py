import os

def get_data_path(data_folder:str="src\\data\\Images", as_annotation:bool=False) -> str:
    sub_dir = "ann_dir" if as_annotation else "img_dir"
    images_dir = f"{data_folder}\\{sub_dir}" 
    return os.path.join(os.getcwd(), images_dir) 