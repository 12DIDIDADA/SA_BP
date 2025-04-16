# 1. Prediction of As content in soil
This is a collection of program scripts based on Python for predicting soil As, which contains four models: RF, SVM, BPNN, and SA_BPNN(Among them, SA_BPNN.log is the output log of SA_BPNN).
# 2. Reproducibility
This section presents information and steps for reproducing our work. Unfortunately, we don't have the permission to share the data.
# 3. Create an environment
This experimental environment is deployed at https://www.suanlix.cn/user/login. It uses Python 3.11 on Ubuntu 20.04 and PyTorch 2.2.2 with CUDA 12.1. (Selecting different graphics cards may result in variations in running time.) For detailed information on deploying cloud servers, refer to (https://docs.suanlix.cn/kuaisurumen.html). Users can install the necessary packages through conda or other methods if they choose deployment environments other than those mentioned above, All library functions used are free, For the detailed installation instructions of Conda, please refer to (https://docs.conda.io/en/latest/)
# 4. Dataset location
The data set should be located in /root/data/, or you can modify the writing in the source code to match your local location. (For example, if your local file is in D:\data.csv, then in the source file, data
= pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv', encoding="gbk") should be modified to data = pd.read_csv(r'D:\data.csv')).The same principle applies to the output saved files.
# 5. Implementation
After creating the environment and saving the data in the correct location, you can complete the training of the models by running RF.py, SVM.py, BPNN.py, and SA_BPNN.py.
