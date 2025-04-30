from typing import Annotated, Any, Sequence, Tuple

from numpydantic import Shape as NPShape
from numpydantic.dtype import Bool, Float64, Int32, Float32
from numpydantic.ndarray import NDArray
from pydantic import Field

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(ge=1)]
UInt = Annotated[int, Field(ge=0)]
Shape = Sequence[PositiveInt]
Shape2D = Tuple[PositiveInt, PositiveInt]
Mask = NDArray[Any, Bool]

RasterWindowShape = NDArray[NPShape["2"], Int32]

# generic raster type
RasterT = NDArray[NPShape["*,*"], Any]
# specific raster types
RasterFloat64 = NDArray[NPShape["*,*"], Float64]
RasterFloat32 = NDArray[NPShape["*,*"], Float32]
RasterInt32 = NDArray[NPShape["*,*"], Int32]
RasterBool = NDArray[NPShape["*,*"], Bool]
