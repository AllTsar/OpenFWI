def txt_path_fixer(txt_path='split_files/curvevel_b_train.txt'):
    #old_prefix = "/projects/piml_inversion/FWIOpenData"
    old_prefix = "/OpenFWIData"
    new_prefix = "OpenFWIData"

    #txt_file = 'split_files/curvevel_b_train.txt'
    with open(txt_path, "r") as f:
        content = f.read()
    new_content = content.replace(old_prefix, new_prefix)
    with open(txt_path, "w") as f:
        f.write(new_content)



def train_val_splitter(txt_path='split_files', ds_name='curvevel_b'):
    with open(f"{txt_path}/{ds_name}_train.txt", "r") as f:
        lines = f.readlines()

    train_lines = lines[:40]
    val_lines = lines[40:]

    with open(f"{txt_path}/{ds_name}_train_new.txt", "w") as f:
        f.writelines(train_lines)

    with open(f"{txt_path}/{ds_name}_val_new.txt", "w") as f:
        f.writelines(val_lines)



if __name__ == "__main__":
    txt_path_fixer(txt_path='split_files/curvevel_b_val.txt')