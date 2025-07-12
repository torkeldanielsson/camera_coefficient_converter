# Camera Coefficient Converter

Converts OpenCV / Kannala-Brandt fisheye camera model parameters to Oden calibration format using non-linear optimization.

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage

```bash
./bin/camera_coefficient_converter example_kb_params.json
```

## Input Format

JSON file with Kannala-Brandt camera parameters:

```json
{
    "cameras": [
        {
            "name": "Left",
            "id": 0,
            "image_width": 1920,
            "image_height": 1080,
            "intrinsics": {
                "fx": 510.6160252953,
                "fy": 510.8831206311,
                "cx": 963.3519332717,
                "cy": 762.5888862920,
                "k1": 0.1391740321,
                "k2": -0.0428092983,
                "k3": 0.0031307063,
                "k4": -0.0000277339
            },
            "extrinsics": {
                "position": [0.1151099, -0.0007408, 0.1109724],
                "rotation_rpy_rad": [0.0, 0.0, 0.0]
            }
        }
    ]
}
```

## Output

Oden view parameters with optimized polynomial coefficients and coordinate system conversion.
Paste it into Oden .vproj file

## License

MIT License (see COPYING.MIT.txt)
