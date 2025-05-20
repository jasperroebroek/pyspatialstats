from typing import Annotated, Any, Sequence, Tuple, Union

from numpydantic import Shape as NPShape
from numpydantic.dtype import Bool, Float64, Int32, UIntP
from numpydantic.ndarray import NDArray
from pydantic import Field

from pyspatialstats.results import (
    BootstrapMeanResult,
    CorrelationResult,
    LinearRegressionResult,
)

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(ge=1)]
UInt = Annotated[int, Field(ge=0)]
Shape = Sequence[PositiveInt]
Shape2D = Tuple[PositiveInt, PositiveInt]
Mask = NDArray[Any, Bool]

RasterWindowShape = NDArray[NPShape["2"], UIntP]

# generic raster type
RasterT = NDArray[NPShape["*,*"], Any]
# specific raster types
RasterFloat64 = NDArray[NPShape["*,*"], Float64]
RasterSizeT = NDArray[NPShape["*,*"], UIntP]
RasterInt32 = NDArray[NPShape["*,*"], Int32]
RasterBool = NDArray[NPShape["*,*"], Bool]

Result = Union[LinearRegressionResult, CorrelationResult, BootstrapMeanResult]
