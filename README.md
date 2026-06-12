# YOLO-CPU — AIx2026 Skeleton

YOLOv2-based object detection inference engine with support for both **full-precision** and **quantized** forward passes. Designed for the AIx2026 hardware accelerator project.

---

## Project Structure

```
skeleton/
├── src/                          # C source code
│   ├── main.c                    # Entry point & FPGA board test harness
│   ├── yolov2_forward_network.c          # Full-precision forward pass
│   ├── yolov2_forward_network_quantized.c # Quantized forward pass
│   ├── box.c / box.h             # Bounding-box utilities
│   └── additionally.c / .h       # Auxiliary helpers
├── bin/                          # Runtime directory
│   ├── aix2024.cfg               # Network configuration
│   ├── aix2024.weights           # Pre-trained weights
│   ├── yolohw.names              # Class labels
│   ├── dataset/                  # Test images & list generator
│   └── script-*.{sh,cmd}         # Test scripts (Unix / Windows)
├── 3rdparty/                     # Third-party libraries (CLBlast, pthreads)
├── include/                      # Header-only dependencies
├── lib/                          # Pre-built static libraries
├── Makefile                      # Unix build
└── yolo_cpu.sln / .vcxproj       # Visual Studio solution
```

---

## Build & Run

### Unix (Linux / macOS)

```bash
cd skeleton
make                              # Compile
cd bin/dataset
python make_list_cur.py           # Generate test-image list
cd ..
```

| Mode | Command |
|------|---------|
| Full-precision (all images) | `sh script-unix-aix2024-test-all.sh` |
| Quantized (all images) | `sh script-unix-aix2024-test-all-quantized.sh` |
| Full-precision (single image) | `sh script-unix-aix2024-test-one.sh` |
| Quantized (single image) | `sh script-unix-aix2024-test-one-quantized.sh` |

### Windows

**Prerequisites:** Visual Studio (tested with VS 2019) and Python.

> If you encounter a version conflict with the `.sln` file:
> 1. Delete `yolo_cpu.sln`
> 2. Double-click `yolo_cpu.vcxproj` to regenerate the solution for your VS version.

```cmd
cd C:\skeleton\bin\dataset
python make_list_cur.py           # Generate test-image list
cd ..
```

| Mode | Command |
|------|---------|
| Full-precision (all images) | `script-wins-aix2024-test-all.cmd` |
| Quantized (all images) | `script-wins-aix2024-test-all-quantized.cmd` |
| Full-precision (single image) | `script-wins-aix2024-test-one.cmd` |
| Quantized (single image) | `script-wins-aix2024-test-one-quantized.cmd` |

---

## FPGA Board Testing

Set `#define FPGA_BOARD_TEST 1` in `src/main.c` to enable the hardware test harness. The current implementation supports Windows only and communicates with the FPGA over a serial/UART interface using the following modes:

| Mode | Value | Description |
|------|-------|-------------|
| `MODE_TEST_HELLO` | `0x01` | Handshake test |
| `MODE_TEST_ECHO` | `0x02` | Echo test |
| `MODE_STORE_RAM` | `0x03` | Write data to FPGA RAM |
| `MODE_LOAD_RAM` | `0x04` | Read data from FPGA RAM |
| `MODE_STORE_CFG` | `0x05` | Send layer configuration |
| `MODE_RUN_ENGINE` | `0x06` | Trigger inference engine |
| `MODE_PAUSE` | `0x07` | Pause execution |

---

## License

See [LICENSE](./LICENSE).
