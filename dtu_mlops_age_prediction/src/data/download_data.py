import opendatasets as od


def download_data(): 
        od.download("https://www.kaggle.com/datasets/frabbisw/facial-age", data_dir = "./dtu_mlops_age_prediction/data/raw",)

download_data()