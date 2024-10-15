from glob import glob
import shutil, os


root      = "/home/hao/data/medical/brats/"
save_root = "/home/hao/data/medical"


modalities = ["t1c", "t1n", "t2w", "t2f"]

print("Processing Brain Data...")

# ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

for TASK in ["GLI", "PED", "SSA"]:
    version=""
    if TASK == "SSA":
        version = "_V2"
    segfiles = sorted(glob(os.path.join(root, f"ASNR-MICCAI-BraTS2023-{TASK}-Challenge-TrainingData{version}/*/*-seg.*")))
    
    print(f"{TASK} Found ", len(segfiles), " files")

    for file in segfiles:
        for modality in modalities:

            imgfile = file.replace("seg",  modality)
            savefilename = os.path.basename(file).replace("-seg", "")


            savepath = os.path.join(save_root, f"Brain/BraTS-{TASK}-{modality.upper()}")
            os.makedirs(savepath, exist_ok=True)

            imgpath = os.path.join(savepath, "img")
            os.makedirs(imgpath, exist_ok=True)
            segpath = os.path.join(savepath, "seg")
            os.makedirs(segpath, exist_ok=True)

            print("from ", imgfile)
            print("save to ", os.path.join(imgpath, savefilename))

            shutil.copy(imgfile, os.path.join(imgpath, savefilename))
            shutil.copy(file, os.path.join(segpath, savefilename))





# ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00117-000/BraTS-GLI-00117-000-t2w.nii.gz