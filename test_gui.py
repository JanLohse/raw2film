import math
import sys
import time
import traceback

import numpy as np
import wgpu
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import QRenderWidget
from wgpu import TextureUsage, get_default_device


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class CpuGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.running = False

    def generate_new(self, width, height, sleep_time=0.5):
        assert not self.running, "Multiple generations triggered at once."
        self.running = True
        time.sleep(sleep_time)
        # Use (height, width, channels) — height first
        image = self.rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        self.running = False
        return image


class GpuGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.running = False
        self.device = get_default_device()
        self.queue = self.device.queue
        self.texture = None
        self.texture_view = None
        self.width = None
        self.height = None
        self.pipeline = None
        self.bind_group = None
        self.seed_buffer = None
        self.seed = 0

        self.setup_shader()
        self.update_seed_buffer()

    def setup_shader(self):
        shader_source = """
        @group(0) @binding(0)
        var outputTex : texture_storage_2d<rgba8unorm, write>;

        @group(0) @binding(1)
        var<uniform> seed : u32;

        fn hash(x: u32) -> u32 {
            var v = x;
            v ^= v >> 16u;
            v *= 0x7feb352du;
            v ^= v >> 15u;
            v *= 0x846ca68bu;
            v ^= v >> 16u;
            return v;
        }

        fn randomFloat(seed: u32) -> f32 {
            return f32(hash(seed)) / 4294967296.0;
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let size = textureDimensions(outputTex);

            if (gid.x >= size.x || gid.y >= size.y) {
                return;
            }

            let pixelIndex = gid.y * size.x + gid.x;

            let base = pixelIndex * 3u;

            let r = randomFloat(base + 0u + seed);
            let g = randomFloat(base + 1u + seed);
            let b = randomFloat(base + 2u + seed);

            textureStore(
                outputTex,
                vec2<i32>(gid.xy),
                vec4<f32>(r, g, b, 1.0)
            );
        }
        """

        shader = self.device.create_shader_module(code=shader_source)

        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": shader,
                "entry_point": "main",
            },
        )

    def setup_bind_group(self, texture):
        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": {"buffer": self.seed_buffer}},
            ],
        )

    def run_shader(self):
        command_encoder = self.device.create_command_encoder()

        compute_pass = command_encoder.begin_compute_pass()

        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.bind_group)

        workgroup_size = 8

        compute_pass.dispatch_workgroups(
            (self.width + workgroup_size - 1) // workgroup_size,
            (self.height + workgroup_size - 1) // workgroup_size,
            1,
        )

        compute_pass.end()

        self.queue.submit([command_encoder.finish()])

    def update_seed_buffer(self):
        self.seed_buffer = self.device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def copy_to_dst(self, src_texture, dst_texture):
        src_w, src_h, _ = src_texture.size
        dst_w, dst_h, _ = dst_texture.size
        offset_x = (dst_w - src_w) // 2
        offset_y = (dst_h - src_h) // 2

        encoder = self.device.create_command_encoder()

        view = dst_texture.create_view()

        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.end()

        encoder.copy_texture_to_texture(
            {
                "texture": src_texture,
                "origin": (0, 0, 0),
            },
            {"texture": dst_texture, "origin": (offset_x, offset_y, 0)},
            (src_w, src_h, 1),
        )

        self.device.queue.submit([encoder.finish()])

    def generate_new(self, width, height, sleep_time=0.5, target_texture=None):
        assert not self.running, "Multiple generations triggered at once."
        self.running = True
        time.sleep(sleep_time)

        self.prepare_texture(width, height)
        self.setup_bind_group(self.texture)

        self.seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)

        self.queue.write_buffer(self.seed_buffer, 0, self.seed.tobytes())
        self.run_shader()

        if target_texture is not None:
            self.copy_to_dst(self.texture, target_texture)

            self.running = False

        else:
            # Read back the texture into an array of uint8 and drop alpha
            image_rgba = self.read_texture(
                self.texture, width, height
            )  # returns uint8 HxWx4
            image_rgb = image_rgba[..., 0:3]  # H x W x 3

            self.running = False
            return image_rgb

    def prepare_texture(self, width, height):
        if self.width == width and self.height == height:
            return

        self.texture = self.device.create_texture(
            size={
                "width": width,
                "height": height,
                "depth_or_array_layers": 1,
            },
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                TextureUsage.STORAGE_BINDING
                | TextureUsage.TEXTURE_BINDING
                | TextureUsage.COPY_SRC
            ),
        )

        self.width = width
        self.height = height

    def read_texture(self, texture, width, height):
        # For an 8-bit RGBA texture, bytes_per_pixel = 4
        bytes_per_pixel = 4
        row_bytes = width * bytes_per_pixel
        # GPU copy-to-buffer often requires rows to be 256-byte-aligned
        padded_row_bytes = ((row_bytes + 255) // 256) * 256
        buffer_size = padded_row_bytes * height

        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        encoder = self.device.create_command_encoder()

        encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": readback_buffer,
                "offset": 0,
                "bytes_per_row": padded_row_bytes,
                "rows_per_image": height,
            },
            {
                "width": width,
                "height": height,
                "depth_or_array_layers": 1,
            },
        )

        self.queue.submit([encoder.finish()])

        readback_buffer.map_sync(wgpu.MapMode.READ)
        data = readback_buffer.read_mapped()
        readback_buffer.unmap()

        # Interpret as uint8
        array = np.frombuffer(data, dtype=np.uint8)

        # Each mapped row has padded_row_bytes bytes; reshape to
        # (height, padded_row_bytes)
        array = array.reshape((height, padded_row_bytes))

        # Now reshape to (height, padded_row_bytes//4, 4) and slice to actual width
        array = array.reshape((height, padded_row_bytes // 4, 4))[:, :width, :]

        return array  # dtype=uint8, shape (height, width, 4)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = QRenderWidget(update_mode="ondemand")
        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))
        self.image_context = None
        self.context_mode = None

        refresh_button = QPushButton("press me")
        refresh_button.setFixedHeight(25)
        refresh_button.pressed.connect(self.trigger_update)

        refresh_slider = QSlider(Qt.Orientation.Horizontal)
        refresh_slider.setFixedHeight(25)
        refresh_slider.valueChanged.connect(self.trigger_update)

        self.gpu_checker = QCheckBox("GPU")
        self.gpu_checker.checkStateChanged.connect(self.trigger_update)

        side_bar = QVBoxLayout()
        side_bar.addWidget(refresh_button)
        side_bar.addWidget(refresh_slider)
        side_bar.addWidget(self.gpu_checker)
        side_bar.addStretch()

        side_bar_widget = QWidget()
        side_bar_widget.setLayout(side_bar)

        self.cpu_generator = CpuGenerator()
        self.gpu_generator = GpuGenerator()

        self.page_splitter = QSplitter()
        self.page_splitter.addWidget(self.image)
        self.page_splitter.addWidget(side_bar_widget)
        self.page_splitter.setStretchFactor(0, 1)
        self.page_splitter.setStretchFactor(1, 0)

        self.setCentralWidget(self.page_splitter)

        self.running = False
        self.waiting = False

        self.threadpool = QThreadPool()

    def start_worker(self, function, semaphore=True, *args, **kwargs):
        if semaphore:
            if self.running:
                self.waiting = True
                return
            else:
                self.running = True
        worker = Worker(function, *args, **kwargs)
        if semaphore:
            worker.signals.finished.connect(self.update_finished)
        self.threadpool.start(worker)

    def update_finished(self):
        """Runs safely on the Main GUI Thread when the worker finishes"""
        self.running = False

        if self.waiting:
            self.waiting = False
            self.trigger_update()

    def trigger_update(self):
        self.create_context()
        self.start_worker(self.update_image)

    def resizeEvent(self, a0):
        self.trigger_update()
        super().resizeEvent(a0)

    def create_context(self):
        target_mode = "wgpu" if self.gpu_checker.isChecked() else "bitmap"
        if self.context_mode == target_mode:
            return
        self.context_mode = target_mode
        old_image = self.image
        self.image = QRenderWidget(update_mode="ondemand")
        self.page_splitter.insertWidget(0, self.image)
        old_image.setParent(None)
        old_image.deleteLater()
        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))
        if self.context_mode == "wgpu":
            self.image_context = self.image.get_wgpu_context()
            self.image_context.configure(
                device=self.gpu_generator.device,
                format=wgpu.TextureFormat.rgba8unorm,
                usage=(TextureUsage.COPY_DST | TextureUsage.RENDER_ATTACHMENT),
            )
        elif self.context_mode == "bitmap":
            self.image_context = self.image.get_bitmap_context()

    def update_image(self, ratio=1.5, **_):
        """Runs inside the background Worker thread"""
        start = time.time()

        full_width, full_height = self.image_context.physical_size

        if full_width <= 0 or full_height <= 0:
            return

        img_width = min(full_width, math.floor(full_height * ratio))
        img_height = min(full_height, math.floor(full_width // ratio))

        if self.context_mode == "wgpu":
            self.gpu_generator.generate_new(
                img_width,
                img_height,
                sleep_time=0.0,
                target_texture=self.image_context.get_current_texture(),
            )
        elif self.context_mode == "bitmap":
            canvas_rgba = np.zeros((full_height, full_width, 4), dtype=np.uint8)

            image = self.cpu_generator.generate_new(
                img_width,
                img_height,
                sleep_time=0.0,
            )
            image = np.ascontiguousarray(image)

            y_offset = (full_height - img_height) // 2
            x_offset = (full_width - img_width) // 2

            canvas_rgba[
                y_offset : y_offset + img_height, x_offset : x_offset + img_width, 0:3
            ] = image
            canvas_rgba[
                y_offset : y_offset + img_height, x_offset : x_offset + img_width, 3
            ] = 255

            self.image_context.set_bitmap(canvas_rgba)

        self.image.request_draw()

        print(f"{time.time() - start:.4f}s")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())
