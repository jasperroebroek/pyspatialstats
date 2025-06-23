from dataclasses import dataclass
from typing import Optional

from pyspatialstats.enums import Uncertainty
from pyspatialstats.focal.result_config.dataclass import FocalDataClassResultConfig
from pyspatialstats.types.results import CorrelationResult, LinearRegressionResult, MeanResult


@dataclass
class FocalMeanResultConfig(FocalDataClassResultConfig):
    uncertainty: Optional[Uncertainty] = None

    @property
    def return_type(self) -> type[MeanResult]:
        return MeanResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        if self.uncertainty == Uncertainty.SE:
            return ('mean', 'se')
        if self.uncertainty == Uncertainty.STD:
            return ('mean', 'std')
        return ('mean',)

    @property
    def cy_fields(self) -> tuple[str, ...]:
        if self.uncertainty == Uncertainty.SE:
            return ('mean', 'se')
        return ('mean',)


@dataclass
class FocalCorrelationResultConfig(FocalDataClassResultConfig):
    p_values: bool = False

    @property
    def return_type(self) -> type[CorrelationResult]:
        return CorrelationResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        fields = tuple()
        for field in self.fields:
            if field == 'p' and not self.p_values:
                continue
            fields += (field,)
        return fields

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.fields if field != 'p')


@dataclass
class FocalLinearRegressionResultConfig(FocalDataClassResultConfig):
    p_values: bool = False

    @property
    def return_type(self) -> type[LinearRegressionResult]:
        return LinearRegressionResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        fields = tuple()
        for field in self.fields:
            if field.startswith('p_') and not self.p_values:
                continue
            fields += (field,)
        return fields

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.fields if not field.startswith('p_'))
