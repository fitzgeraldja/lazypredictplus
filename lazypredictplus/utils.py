import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def basic_model_pipeline(
    model, num_scaler="robust", cat_encoder="ohe", imputer="simple"
):
    """From given sklearn style model make simple pipeline, along with suitable transformers

    Args:
        model (_type_): _description_
        num_scaler (str, optional): _description_. Defaults to 'robust'.
        cat_encoder (str, optional): _description_. Defaults to 'ohe'.
        imputer (str, optional): _description_. Defaults to 'simple'.
    """

    col_transformer = make_column_transformer(
        (RobustScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_exclude=np.number)),
    )
    pipeline = make_pipeline(SimpleImputer(), col_transformer, model)
    return pipeline
