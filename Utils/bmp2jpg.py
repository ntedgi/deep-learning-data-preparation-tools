import os
from PIL import Image
from PIL import ImageFile

ImageFile.MAXBLOCK = 2 ** 20


def convertBMP2JPG(dir, file):
    oldFile = os.path.join(dir, file)
    assert (oldFile.endswith('.bmp'))
    newFile = oldFile[0:len(oldFile) - 3] + 'jpg'
    print("%s to %s" % (oldFile, newFile))
    I = Image.open(oldFile)
    I.save(newFile, "JPEG", quality=80, optimize=True, progressive=True)
    os.remove(oldFile)


def convertBMPs2JPGs(dir):
    bmpFiles = [f for f in os.listdir(dir) if f.endswith('.bmp')]
    print("%s: %d .bmp to .jpg" % (dir, len(bmpFiles)))
    for b in bmpFiles:
        convertBMP2JPG(dir, b)


import sys


def main():
    cwd = os.getcwd()
    if len(sys.argv) > 1:
        for argi in range(1, len(sys.argv)):
            print(sys.argv[argi])
            path = os.path.join(cwd, sys.argv[argi])
            if os.path.isdir(path):
                convertBMPs2JPGs(path)
            else:
                print("error: %s is not a directory" % (path))
    else:
        dir = os.getcwd()
        subdirs = [d for d in os.listdir(dir) if os.path.isdir(d)]
        for d in subdirs:
            path = os.path.join(cwd, d)
            convertBMPs2JPGs(path)


if __name__ == "__main__":
    main()
