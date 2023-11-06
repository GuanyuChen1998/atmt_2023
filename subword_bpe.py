import os

num_operations = 20000
size = 4000
order = 100
data_folder = "data/en-fr/"

# Raw Folder
train_src = os.path.join(data_folder, "raw/train.fr")
train_tgt = os.path.join(data_folder, "raw/train.en")
test_src = os.path.join(data_folder, "raw/test.fr")
test_tgt = os.path.join(data_folder, "raw/test.en")
valid_src = os.path.join(data_folder, "raw/valid.fr")
valid_tgt = os.path.join(data_folder, "raw/valid.en")
tiny_train_src = os.path.join(data_folder, "raw/tiny_train.fr")
tiny_train_tgt = os.path.join(data_folder, "raw/tiny_train.en")

# Processed Folder
p_train_src = os.path.join(data_folder, "prepared/train.fr")
p_train_tgt = os.path.join(data_folder, "prepared/train.en")
p_test_src = os.path.join(data_folder, "prepared/test.fr")
p_test_tgt = os.path.join(data_folder, "prepared/test.en")
p_valid_src = os.path.join(data_folder, "prepared/valid.fr")
p_valid_tgt = os.path.join(data_folder, "prepared/valid.en")
p_tiny_train_src = os.path.join(data_folder, "prepared/tiny_train.fr")
p_tiny_train_tgt = os.path.join(data_folder, "prepared/tiny_train.en")

codes_file = os.path.join(data_folder, "prepared/bpe.fr.en")
dict_src = os.path.join(data_folder, "prepared/dict.fr")
dict_tgt = os.path.join(data_folder, "prepared/dict.en")


os.system(f'subword-nmt learn-joint-bpe-and-vocab --input {train_src} {train_tgt} -s {num_operations} -o {codes_file} --write-vocabulary {dict_src} {dict_tgt}')
os.system(f'subword-nmt apply-bpe --dropout 0.1 -c {codes_file} --vocabulary {dict_src} --vocabulary-threshold 1 < {train_src} > {p_train_src}')
os.system(f'subword-nmt apply-bpe --dropout 0.1 -c {codes_file} --vocabulary {dict_tgt} --vocabulary-threshold 1 < {train_tgt} > {p_train_tgt}')

os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_src} --vocabulary-threshold 1 < {test_src} > {p_test_src}')
os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_tgt} --vocabulary-threshold 1 < {test_tgt} > {p_test_tgt}')

os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_src} --vocabulary-threshold 1 < {valid_src} > {p_valid_src}')
os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_tgt} --vocabulary-threshold 1 < {valid_tgt} > {p_valid_tgt}')

os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_src} --vocabulary-threshold 1 < {tiny_train_src} > {p_tiny_train_src}')
os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {dict_tgt} --vocabulary-threshold 1 < {tiny_train_tgt} > {p_tiny_train_tgt}')
