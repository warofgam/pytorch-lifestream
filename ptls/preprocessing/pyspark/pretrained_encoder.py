from itertools import chain

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pyspark.col_transformer import ColTransformerPysparkMixin


class PretrainedEncoder(ColTransformerPysparkMixin, ColCategoryTransformer):
    def __init__(self,
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 max_cat_num: int = 10000,
                 pretrained_dict = {},
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )

        self.mapping = None
        self.other_values_code = None
        self.max_cat_num = max_cat_num
        self.pretrained_dict = pretrained_dict

    def get_col(self, x: pyspark.sql.DataFrame):
        return x.withColumn(self.col_name_target,
                          F.coalesce(F.col(self.col_name_original).cast('string'), F.lit('#EMPTY')))

    @property
    def dictionary_size(self):
        return self.other_values_code + 1

    def transform(self, x: pyspark.sql.DataFrame):
        df = self.get_col(x)
        df_encoder = df.groupby(self.col_name_target).agg(F.count(F.lit(1)).alias('_cnt'))
        self.mapping = {row[self.col_name_target]: self.pretrained_dict[row[self.col_name_target]] for row in df_encoder.collect()}
        mapping_expr = F.create_map([F.lit(x) for x in chain(*self.mapping.items())])
        df = df.withColumn(self.col_name_target, mapping_expr[F.col(self.col_name_target)])
        df = df.fillna(value=self.other_values_code, subset=[self.col_name_target])

        x = super().transform(df)
        return x
