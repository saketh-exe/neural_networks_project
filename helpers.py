import os 
def bulk_rename(path):
    try:
        items = os.listdir(path)

        # filtering 
        files = [file for file in items if os.path.isfile(os.path.join(path,file))]
        
        files.sort()

        for index,name in enumerate(files,start=1):
            ext = name.split(".")[-1]
            new_name = f"{index}-Image.{ext}"
            src = os.path.join(path,name)
            dest = os.path.join(path,new_name)

            if not os.path.exists(dest):
                os.rename(src,dest)
                print(f"Renamed {src} to {dest}")
            else:
                print("skipped")
    except:
        print("Error")


bulk_rename("./images/raw_train/")

