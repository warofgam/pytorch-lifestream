import pandas as pd

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class PretrainedEncoder(ColTransformerPandasMixin, ColCategoryTransformer):
    def __init__(self,
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 pretrained_dict = {}
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )
        self.pretrained_dict = pretrained_dict
    @property
    def dictionary_size(self):
        return self.other_values_code + 1

    def transform(self, x: pd.DataFrame):
        #pd_col = x[self.col_name_original].astype(str)
        pd_col = x[self.col_name_original]
        x = self.attach_column(x, pd_col.map(self.pretrained_dict).rename(self.col_name_target))
        x = super().transform(x)
        return x
