import pandas as pd
import os
from tqdm import tqdm
import random
import shutil

os.rename("CelebA/img_align_celeba", "CelebA/img_align_celeba_train")
os.mkdir("CelebA/img_align_celeba_test")

source = 'CelebA/img_align_celeba_train'
dest = 'CelebA/img_align_celeba_test'
files = os.listdir(source)
no_of_files = 20000

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)

df = pd.read_table("celebA/Anno/list_attr_celeba.txt", skiprows=1, delim_whitespace=True)
names_list = os.listdir("CelebA/img_align_celeba_test/")
df_test = df[df.index.isin(names_list)]
df_test.to_csv("CelebA/Anno/list_attr_celeba_test.txt", sep=" ", header=True)

for i in tqdm(names_list):
    df.drop([i], inplace=True)

df.to_csv("CelebA/Anno/list_attr_celeba_train.txt", sep=" ", header=True)