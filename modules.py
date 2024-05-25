def data_preprocessing(input_path: str, output_path: str, file_name: str=''):
    
    """Чтение и формирование данных
    
    Arguments:
        input_path (str): путь к входным данным
        output_path (str): путь к выходным данным
        file_name (str): название файла

    Returns:
        X: матрица 'объект-признак'
        y: вектор целевой переменной
    """
    
    from pandas import read_parquet, to_datetime, to_timedelta
    import os
    from pathlib import Path
    
    normalized_input_path = os.path.normpath(input_path)
    in_path = Path(normalized_input_path)
    if not in_path.exists():
        raise IsADirectoryError(f"Переданный путь '{input_path}' не существует")
    if not in_path.is_file():
        raise IsADirectoryError(f"Переданный путь '{input_path}' не является файлом")
    
    data = read_parquet(input_path)

    data["operating_day"] = to_datetime(data["operating_day"], format="mixed")
    data["time"] = data["operating_day"] + to_timedelta(data["arrival"], unit='s')
    data = data.drop(columns="operating_day")
    data = data[data["time"].dt.year >= 2022]
    data = data.drop_duplicates()
    data = data.dropna()
    
    data = data[data["vehicle_seats"] <= 40]
    data = data[data["passengers"] <= 50]
    
    data = data[["time", "line_id", "stop_id", "passengers"]]
    data["hour"] = data["time"].dt.hour
    data["minute"] = data["time"].dt.minute
    
    data = data.sort_values(by="time")
    
    if not file_name:
        file_name = "data"
    
    normalized_output_path = os.path.normpath(output_path)
    out_path = Path(normalized_output_path)

    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    
    data = data.reindex(columns=["time", "line_id", "stop_id", "hour", "minute", "passengers"])
    data.to_csv(output_path+file_name+".csv", index=False)


def train_model(dataframe, input_path: str):
    
    """Обучение модели
    
    Arguments:
        dataframe (DataFrame): исходный набор данных
        input_path (str): путь для сохранения файлов

    Return:
        model: модель случайного леса
    """
    
    from sklearn.ensemble import RandomForestRegressor
    import pickle
    import os
    from pathlib import Path
    
    X = dataframe.drop(columns=["time", "passengers"])
    y = dataframe["passengers"]
    
    model = RandomForestRegressor(max_depth=16).fit(X, y)

    normalized_path = os.path.normpath(input_path)
    path = Path(normalized_path)
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    with open(path / "model.pkl", "wb") as file:
        pickle.dump(model, file)


def random_forest_predict(path_to_model: str, **kwargs) -> float:
    
    """Получение прогноза модели RandomForest
    
    Argument:
        path_to_model (str): путь к модели RandomForest

    Return:
        predict (float): прогноз
    """
    
    import pickle
    from pandas import DataFrame
    import os
    from pathlib import Path

    normalized_path = os.path.normpath(path_to_model)
    path = Path(normalized_path)
    
    if not path.exists():
        raise IsADirectoryError(f"Переданный путь '{path_to_model}' не существует")
    
    if not path.is_file():
        raise IsADirectoryError(f"Переданный путь '{path_to_model}' не является файлом")
    
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
    
    if "n_estimators" in model.get_params():
        return model.predict(DataFrame(kwargs, index=[0]))
    print("Модель не является RandomForest")
    return -1


def ema_predict(dataframe, **kwargs) -> float:
    
    """Получение прогноза модели скользящего среднего
    
    Argument:
        dataframe: исходный набор данных
    
    Return:
        predict (float): прогноз
    
    """
    
    from pandas import DataFrame, concat
    
    dataset = DataFrame(columns=["time", "passengers"])
    
    for delta in range(kwargs["radius"], 0, -1):
        m = kwargs["minute"] - delta
        h = kwargs["hour"]
        if m < 0:
            m = m + 60
            h = h - 1
            if h < 0:
                h = h + 24
        temp = dataframe[(dataframe["hour"] == h) & (dataframe["minute"] == m)]
        if not temp.empty:
            dataset = concat([dataset, temp[["time", "passengers"]]], axis=0)
    
    for delta in range(kwargs["radius"]+1):
        m = kwargs["minute"] + delta
        h = kwargs["hour"]
        if m >= 60:
            m = m % 60
            h = h + 1
            if h >= 24:
                h = h % 24
        temp = dataframe[(dataframe["hour"] == h) & (dataframe["minute"] == m)]
        if not temp.empty:
            dataset = concat([dataset, temp[["time", "passengers"]]], axis=0)
    
    if len(dataset) < 1:
        return -1
    dataset = dataset.sort_values(by="time")
    return dataset["passengers"].ewm(alpha=kwargs["alpha"]).mean().iat[-1]


def get_predict(dataframe, path_to_models: str, is_random_forest=True, is_ema=True, output_path: str="./", file_name: str='') -> float:
    
    """Получение прогноза
    
    Arguments:
        dataframe: набор данных
        path_to_models (str): путь к обученным моделям
        is_random_forest (bool): использование модели RandomForest
        is_ema (bool): использование модели EMA
        output_path (str): путь файла с результатами
        file_name (str): название выходного файла
    
    Return:
        predict (float): прогноз
    """
    
    from pandas import DataFrame
    import os
    from pathlib import Path
    
    result = []
    
    line_to_stops = {}
    for label in dataframe["line_id"].unique():
        line_to_stops[label] = dataframe[dataframe["line_id"] == label]["stop_id"].unique()
    
    for line_id in line_to_stops:
        line_data = dataframe[dataframe["line_id"] == line_id]
        for stop_id in line_to_stops[line_id]:
            stop_data = line_data[line_data["stop_id"] == stop_id]
            for hour in range(5, 24):
                for minute in range(0, 60, 5):
                    
                    if is_random_forest:
                        random_forest_result = random_forest_predict(path_to_models, line_id=line_id, stop_id=stop_id, hour=hour, minute=minute).item()
                    else:
                        random_forest_result = -1
                    
                    if is_ema:
                        ema_result = ema_predict(stop_data, radius=1, alpha=0.5, hour=hour, minute=minute)
                    else:
                        ema_result = -1
                    
                    if (random_forest_result >= 0) and (ema_result >= 0):
                        predict = round((random_forest_result + ema_result) / 2)
                    elif random_forest_result == -1:
                        if ema_result == -1:
                            predict = 0
                        else:
                            predict = round(ema_result)
                    else:
                        predict = round(random_forest_result)
                        
                    row = {"line_id": line_id, "stop_id": stop_id, "hour": hour, "minute": minute, "passengers": predict}
                    result.append(row)
    
    normalized_path = os.path.normpath(output_path)
    path = Path(normalized_path)
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    if not file_name:
        file_name = "result"
    
    DataFrame(result).to_csv(path.joinpath(path, file_name+".csv"), index=False)