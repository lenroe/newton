#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Hydroelastic Collision Benchmark
#
# Benchmarks hydroelastic collision performance across different scenes and solvers.
#
# Usage: uv run --with pynvml hydro_bench.py --all

import argparse
import functools
import gc
import itertools
import time
from dataclasses import dataclass

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
import newton.usd
from newton._src.utils.download_assets import download_git_folder
from newton.examples.robot.example_robot_panda_hydro import Example as PandaHydroExample

HYDROELASTIC = True

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    device = wp.get_device()
    if not device.is_cuda:
        return 0.0
    
    try:
        # Try using pynvml for accurate memory reporting
        import pynvml
        pynvml.nvmlInit()
        # Extract device index from device name (e.g., "cuda:0" -> 0)
        device_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)  # Convert to MB
    except ImportError:
        # Fallback: use warp's allocator info if available
        try:
            allocator = wp.get_device().allocator
            if hasattr(allocator, 'bytes_used'):
                return allocator.bytes_used / (1024 * 1024)
        except Exception:
            pass
        return -1.0  # Unknown

# ============================================================================
# Configuration
# ============================================================================

SCENE_WARMUP_FRAMES = {
    "nut_bolt": 1000,
    "bunny_pyramid": 200,
    "panda_hydro": 200,
}

SCENE_BENCHMARK_FRAMES = {
    "nut_bolt": 200,
    "bunny_pyramid": 200,
    "panda_hydro": 400,
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    scene: str  # "nut_bolt", "bunny_pyramid", or "panda_hydro"
    solver: str  # "xpbd" or "mujoco"
    num_worlds: int  # Number of parallel worlds
    warmup_frames: int | None = None  # Per-scene default if None
    benchmark_frames: int | None = None  # Per-scene default if None
    fps: int = 120  # Target frames per second
    sim_substeps: int = 2  # Simulation substeps per frame
    visualize: bool = False  # Use GL viewer instead of headless
    reduce_contacts: bool = True  # Enable contact reduction in collision pipeline
    sdf_hydroelastic_config: newton.geometry.HydroelasticSDF.Config | None = None  # Per-scene hydroelastic config


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    scene: str
    solver: str
    num_worlds: int
    reduce_contacts: bool  # Whether contact reduction was enabled
    total_sim_time: float  # Total simulated time (seconds)
    wall_time: float  # Wall clock time (seconds)
    real_time_factor: float  # How many times faster than real-time
    avg_frame_time_ms: float  # Average time per frame (milliseconds)
    frames: int  # Number of frames benchmarked
    vram_mb: float  # GPU memory usage in MB
    median_contacts: int  # Median number of rigid contacts per frame


# ============================================================================
# Shape Configurations
# ============================================================================

# Nut-bolt scene shape config (hydroelastic, high resolution SDF)
NUT_BOLT_SDF_MAX_RESOLUTION = 128
NUT_BOLT_SDF_NARROW_BAND_RANGE = (-0.001, 0.001)

NUT_BOLT_SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,
    kd=1e4,
    gap=0.002,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=HYDROELASTIC,
)

# Bunny pile scene shape config (hydroelastic mesh)
BUNNY_SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    mu=1.0,
    margin=0.0,
    density=1000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=HYDROELASTIC,
)

# ============================================================================
# Scene Builders
# ============================================================================

ASSEMBLY_STR = "m20_tight"

@functools.lru_cache(maxsize=1)
def _get_nut_bolt_assets():
    """Download nut/bolt assets, caching the result across calls."""
    repo_url = "https://github.com/isaac-sim/IsaacGymEnvs.git"
    print(f"Downloading nut/bolt assets from {repo_url}...")
    asset_path = download_git_folder(repo_url, "assets/factory/mesh/factory_nut_bolt")
    print(f"Assets downloaded to: {asset_path}")
    return asset_path


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh_file: str,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    label: str | None = None,
    center_origin: bool = True,
    scale: float = 1.0,
) -> int:
    """Add a mesh object to the model builder."""
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2
        vertices = vertices - center
        center_vec = wp.vec3(center) * float(scale)
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=NUT_BOLT_SDF_MAX_RESOLUTION,
        narrow_band_range=NUT_BOLT_SDF_NARROW_BAND_RANGE,
        margin=shape_cfg.gap,
        scale=(scale, scale, scale),
    )
    body = builder.add_body(label=label, xform=transform)
    builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


def build_nut_bolt_world(scene_scale: float = 1.0) -> newton.ModelBuilder:
    """Build a single nut-bolt assembly world."""
    asset_path = _get_nut_bolt_assets()

    world_builder = newton.ModelBuilder()
    world_builder.default_shape_cfg.gap = 0.001 * scene_scale

    bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
    nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")

    # Add bolt at origin
    bolt_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    add_mesh_object(
        world_builder,
        bolt_file,
        bolt_xform,
        NUT_BOLT_SHAPE_CFG,
        label="bolt",
        center_origin=True,
        scale=scene_scale,
    )

    # Add nut above bolt
    nut_xform = wp.transform(
        wp.vec3(0.0, 0.0, 0.041 * scene_scale),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
    )
    add_mesh_object(
        world_builder,
        nut_file,
        nut_xform,
        NUT_BOLT_SHAPE_CFG,
        label="nut",
        center_origin=True,
        scale=scene_scale,
    )

    return world_builder


BUNNY_SDF_MAX_RESOLUTION = 64
BUNNY_SDF_NARROW_BAND = 0.002
BUNNY_SCALE_TARGET = 0.08  # Target extent of each bunny [m]
BUNNY_PYRAMID_BASE = 6
BUNNY_SPACING_FACTOR = 1.05  # Multiplier on bunny extent for spacing


@functools.lru_cache(maxsize=1)
def _get_bunny_mesh():
    """Load the Stanford Bunny mesh from bundled assets (cached)."""
    from pxr import Usd

    bunny_usd = newton.examples.get_asset("bunny.usd")
    usd_stage = Usd.Stage.Open(str(bunny_usd))
    mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))
    return mesh


def build_bunny_pyramid_world() -> newton.ModelBuilder:
    """Build a pyramid of bunny meshes (5x5 base) under gravity."""
    mesh = _get_bunny_mesh()

    # Compute scale so the bunny fits in BUNNY_SCALE_TARGET
    verts = np.array(mesh.vertices)
    extent = verts.max(axis=0) - verts.min(axis=0)
    scale = BUNNY_SCALE_TARGET / float(max(extent))
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0

    # Build SDF once at the target scale
    mesh.build_sdf(
        max_resolution=BUNNY_SDF_MAX_RESOLUTION,
        narrow_band_range=(-BUNNY_SDF_NARROW_BAND, BUNNY_SDF_NARROW_BAND),
        margin=BUNNY_SDF_NARROW_BAND,
        scale=(scale, scale, scale),
    )

    spacing = BUNNY_SCALE_TARGET * BUNNY_SPACING_FACTOR
    base = BUNNY_PYRAMID_BASE

    world_builder = newton.ModelBuilder()

    for level in range(base):
        n = base - level  # number of bunnies per side at this level
        z = spacing * 0.5 + level * spacing
        # Center each layer
        offset = (n - 1) * spacing * 0.5
        for ix in range(n):
            for iy in range(n):
                x = ix * spacing - offset
                y = iy * spacing - offset
                # Offset position so mesh center aligns with grid point
                pos = wp.vec3(
                    x - float(center[0]) * scale,
                    y - float(center[1]) * scale,
                    z - float(center[2]) * scale,
                )
                body = world_builder.add_body(
                    xform=wp.transform(pos, wp.quat_identity()),
                    label=f"bunny_{level}_{ix}_{iy}",
                )
                world_builder.add_shape_mesh(
                    body,
                    mesh=mesh,
                    scale=wp.vec3(scale, scale, scale),
                    cfg=BUNNY_SHAPE_CFG,
                )

    return world_builder


# ============================================================================
# Panda Hydro (Example-based scene)
# ============================================================================

class PandaHydroScene:
    """Wraps the panda_hydro Example as a benchmark scene."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        args = argparse.Namespace(
            scene="pen",
            test=False,
            world_count=config.num_worlds,
        )
        if config.visualize:
            viewer = newton.viewer.ViewerGL()
        else:
            viewer = newton.viewer.ViewerNull()
        self.example = PandaHydroExample(viewer, args)

        self.frame_dt = self.example.frame_dt
        self.contacts = self.example.contacts

    def step(self):
        self.example.step()
        if self.config.visualize:
            self.example.render()


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkScene:
    """A benchmark scene that can be stepped and timed."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.fps = config.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = config.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Build scene based on type
        if config.scene == "nut_bolt":
            world_builder = build_nut_bolt_world(scene_scale=1.0)
            self.ground_offset = -0.01
        elif config.scene == "bunny_pyramid":
            world_builder = build_bunny_pyramid_world()
            self.ground_offset = 0.0
        else:
            raise ValueError(f"Unknown scene: {config.scene}")

        # Create main scene with ground plane and replicated worlds
        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.gap = 0.001
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_offset),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=config.num_worlds)

        self.model = main_scene.finalize()

        iso_mult = 1
        if config.scene == "nut_bolt":
            max_con_per_world = 450 if config.reduce_contacts else 300000
        elif config.scene == "bunny_pyramid":
            max_con_per_world = 14000 if config.reduce_contacts else 1_800_000
            iso_mult = 2
        else:
            raise ValueError(f"Unknown scene: {config.scene}")

        hydro_cfg = newton.geometry.HydroelasticSDF.Config(
            buffer_mult_iso=iso_mult, buffer_mult_contact=1, moment_matching=True, reduce_contacts=config.reduce_contacts,
        )
        
        rigid_contact_max = max_con_per_world * config.num_worlds

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=config.reduce_contacts,
            broad_phase="sap" if config.scene == "bunny_pyramid" else "explicit",
            sdf_hydroelastic_config=hydro_cfg,
            rigid_contact_max=rigid_contact_max,
        )

        # Create solver
        if config.solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=10,
                rigid_contact_relaxation=0.8,
            )
        elif config.solver == "mujoco":

            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=max_con_per_world,
                nconmax=max_con_per_world,
                iterations=15,
                ls_iterations=50,
                impratio=1.0,
            )
        else:
            raise ValueError(f"Unknown solver: {config.solver}")

        # Initialize states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Create viewer (GL for visualization, Null for headless)
        if config.visualize:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
            self.viewer.log_state(self.state_0)
            self.viewer.set_world_offsets((0.1, 0.1, 0.0))
        else:
            self.viewer = newton.viewer.ViewerNull()

        # Try to capture CUDA graph for faster execution
        self.graph = None
        if wp.get_device().is_cuda:
            try:
                with wp.ScopedCapture() as capture:
                    self._simulate_frame()
                self.graph = capture.graph
            except Exception as e:
                print(f"Warning: Could not capture CUDA graph: {e}")
                self.graph = None

    def _simulate_frame(self):
        """Simulate one frame (called during graph capture and execution)."""
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Step the simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_frame()
        self.sim_time += self.frame_dt

        # Update viewer if visualizing
        if self.config.visualize:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a benchmark with the given configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: scene={config.scene}, solver={config.solver}, num_worlds={config.num_worlds}")
    print(f"{'='*60}")

    # Create scene
    print("Building scene...")
    if config.scene == "panda_hydro":
        scene = PandaHydroScene(config)
    else:
        scene = BenchmarkScene(config)
    
    # Measure VRAM after scene creation
    wp.synchronize()
    vram_mb = get_gpu_memory_mb()
    if vram_mb >= 0:
        print(f"VRAM usage: {vram_mb:.1f} MB")
    else:
        print("VRAM usage: unknown (pynvml not available)")

    # Resolve per-scene defaults
    warmup = config.warmup_frames if config.warmup_frames is not None else SCENE_WARMUP_FRAMES.get(config.scene, 500)
    bench_frames = config.benchmark_frames if config.benchmark_frames is not None else SCENE_BENCHMARK_FRAMES.get(config.scene, 200)

    # Warmup
    print(f"Warming up ({warmup} frames)...")
    for _ in range(warmup):
        scene.step()

    # Synchronize before timing
    wp.synchronize()

    # Pass 1: timing (no device sync between frames)
    print(f"Benchmarking ({bench_frames} frames)...")
    start_time = time.perf_counter()
    for _ in range(bench_frames):
        scene.step()
    wp.synchronize()
    end_time = time.perf_counter()

    # Pass 2: contact counting (syncs each frame to read contact count)
    contact_count_frames = 100
    print(f"Counting contacts ({contact_count_frames} frames)...")
    contact_counts = []
    for _ in range(contact_count_frames):
        scene.step()
        contact_counts.append(int(scene.contacts.rigid_contact_count.numpy()[0]))

    # Calculate results
    wall_time = end_time - start_time
    total_sim_time = bench_frames * scene.frame_dt * config.num_worlds
    real_time_factor = total_sim_time / wall_time
    avg_frame_time_ms = (wall_time / bench_frames) * 1000
    median_contacts = int(np.median(contact_counts))

    result = BenchmarkResult(
        scene=config.scene,
        solver=config.solver,
        num_worlds=config.num_worlds,
        reduce_contacts=config.reduce_contacts,
        total_sim_time=total_sim_time,
        wall_time=wall_time,
        real_time_factor=real_time_factor,
        avg_frame_time_ms=avg_frame_time_ms,
        frames=bench_frames,
        vram_mb=vram_mb,
        median_contacts=median_contacts,
    )

    print("\nResults:")
    print(f"  Wall time: {wall_time:.3f} s")
    print(f"  Simulated time (aggregate): {total_sim_time:.3f} s")
    print(f"  Real-time factor: {real_time_factor:.2f}x")
    print(f"  Avg frame time: {avg_frame_time_ms:.2f} ms")
    print(f"  Median contacts: {median_contacts}")
    print(f"  VRAM: {vram_mb:.1f} MB" if vram_mb >= 0 else "  VRAM: unknown")

    return result


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table of all benchmark results."""
    print("\n" + "="*110)
    print("BENCHMARK SUMMARY")
    print("="*110)
    print(f"{'Scene':<12} {'Solver':<8} {'Worlds':<8} {'Reduce':<8} {'RT Factor':<12} {'Frame Time':<12} {'Contacts':<12} {'VRAM':<12}")
    print("-"*110)

    for r in results:
        vram_str = f"{r.vram_mb:.1f} MB" if r.vram_mb >= 0 else "N/A"
        reduce_str = "on" if r.reduce_contacts else "off"
        print(f"{r.scene:<12} {r.solver:<8} {r.num_worlds:<8} {reduce_str:<8} {r.real_time_factor:>10.2f}x {r.avg_frame_time_ms:>10.2f}ms {r.median_contacts:>10} {vram_str:>10}")

    print("="*110)


def main():
    parser = argparse.ArgumentParser(description="Hydroelastic Collision Benchmark")
    parser.add_argument(
        "--scene",
        type=str,
        choices=["nut_bolt", "bunny_pyramid", "panda_hydro"],
        default="nut_bolt",
        help="Scene to benchmark",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["xpbd", "mujoco"],
        default="mujoco",
        help="Solver to use",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help="Number of parallel worlds",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help="Number of warmup frames (default: per-scene — nut_bolt=1000, bunny_pyramid=500, panda_hydro=200)",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=None,
        help="Number of frames to benchmark (default: per-scene, currently 200 for all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark configurations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Use GL viewer for visualization (default is headless with ViewerNull)",
    )
    parser.add_argument(
        "--no-reduce-contacts",
        action="store_true",
        help="Disable contact reduction in the collision pipeline",
    )

    args = parser.parse_args()

    wp.set_device(args.device)
    print(f"Using device: {wp.get_device()}")

    results = []

    if args.all:

        scenes = ["nut_bolt", "bunny_pyramid", "panda_hydro"]
        solvers = ["mujoco", "xpbd"]
        num_worlds = [1, 16, 64, 128, 512, 1024]
        reduce_options = [True, False]

        for scene, solver, n_worlds, reduce in itertools.product(scenes, solvers, num_worlds, reduce_options):
            if n_worlds > 1 and scene == "bunny_pyramid":
                continue

            if scene == "bunny_pyramid" and solver == "mujoco":
                continue

            # panda_hydro uses its own MuJoCo solver; skip xpbd variant
            if scene == "panda_hydro" and (solver == "xpbd" or reduce == False):
                continue

            
            config = BenchmarkConfig(
                scene=scene,
                solver=solver,
                num_worlds=n_worlds,
                warmup_frames=args.warmup_frames,
                benchmark_frames=args.benchmark_frames,
                visualize=args.visualize,
                reduce_contacts=reduce,
            )
            try:
                result = run_benchmark(config)
                results.append(result)
            except Exception as e:
                print(f"\nERROR: Benchmark failed for scene={scene}, solver={solver}, "
                      f"num_worlds={n_worlds}, reduce={reduce}: {e}")
            # Clear VRAM between runs to avoid OOM with large configs
            gc.collect()
            wp.synchronize()
    else:
        # Run single configuration
        config = BenchmarkConfig(
            scene=args.scene,
            solver=args.solver,
            num_worlds=args.num_worlds,
            warmup_frames=args.warmup_frames,
            benchmark_frames=args.benchmark_frames,
            visualize=args.visualize,
            reduce_contacts=not args.no_reduce_contacts,
        )
        result = run_benchmark(config)
        results.append(result)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()