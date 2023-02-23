import os,cv2,pathlib
pathh=input("enter the path of the folder")
os.chdir(pathh)
print(os.getcwd())
list_imges=(os.listdir())
try:
    os.makedirs(os.path.join("flip"))
except:
    pass
print(os.getcwd())
for img in list_imges:
    if img.endswith(".jpg"):
        print(img)
        cap=cv2.imread(img)
        cv2.imwrite("flip_{}".format(img),cv2.flip(cap,flipCode=1))
        pathlib.Path(os.getcwd()+r"\flip_{}".format(img)).rename(os.getcwd()+r"\flip\flip_{}".format(img))


