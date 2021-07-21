import os.path
import pickle
from skimage.io import imsave


print("BG")

with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)

    print(picture_BG)


print("SP")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)

# look at the data of number two and visualise it
    print(picture_SP.keys())

for fname in ['data/PAML_data/Q2_BG_dict.pkl',  'data/PAML_data/Q2_SP_dict.pkl']:
    print("data/PAML_data/Q2_BG_dict.pkl", fname)
    with open(fname,'rb') as fp:
        data = pickle.load(fp)
        print("The sets in the dictionary are:", data.keys())
        print("The number of entries for each set is:", len(data['train']), len(data['validation']), len(data['evaluation']))
        print("The size of each entry is:", data['train'][0].shape, data['validation'][0].shape, data['evaluation'][0].shape)
        export_path = "data/data_assignment/train"
        for i, img in enumerate(data['train']):
            imsave(os.path.join(export_path, "img_{}.png".format(i)), img)
        export_path = "data/data_assignment/validation"
        for i, img in enumerate(data['validation']):
            imsave(os.path.join(export_path, "img_{}.png".format(i)), img)

