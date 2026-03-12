# COPC-RS

A Rust CLI tool that converts LAZ/LAS point cloud files into [COPC (Cloud-Optimized Point Cloud)](https://copcio.github.io) format. The [COPC 1.0 spec](https://github.com/copcio/copcio.github.io/blob/main/copc-specification-1.0.pdf) defines COPC as a LAZ 1.4 file with an octree-based spatial index.

## Building and running

```bash
cargo build --release
./target/release/copc_rs <input.laz ...> -o output.copc.laz
./target/release/copc_rs <directory/> -o output.copc.laz
```

### CLI flags

- `--memory-limit <SIZE>` – Maximum memory budget (e.g. `16G`, `8G`, `4096M`, `512M`). Default: `16G`. A 0.75x safety factor is applied internally.
- `--temp-dir <PATH>` – Temp directory for intermediate files. Default: system temp directory.

## Architecture

The converter runs in multiple passes to keep memory usage bounded:

1. **Pass 1 – Scan** (`octree::OctreeBuilder::scan`): Reads headers of all input files in parallel (rayon) to collect the global bounding box, total point count, and scale/offset transforms. Determines octree depth so leaf nodes average ≤ 65K points.

2. **Pass 2 – Distribute** (`octree::OctreeBuilder::distribute`): Reads every point, assigns it to a leaf voxel key, and flushes to per-key temp files on disk. Uses `read_all_points_into` (fast parallel decompression) when a file fits in half the memory budget, otherwise falls back to batched `read_points_into` to stay within bounds. Flush interval is scaled from the memory budget.

3. **Build node map** (`octree::OctreeBuilder::build_node_map`): Processes bottom-up one level at a time with disk offloading. For each level, parents are processed in parallel (rayon): children's points are loaded from temp files, stride-sampled, and the parent sample is written to a new temp file. Only one level's worth of data is in memory at a time. Returns `Vec<(VoxelKey, point_count)>`.

4. **Write COPC** (`writer::write_copc`): Streams compressed data directly to the output file via `LasZipCompressor<BufWriter<File>>`, reading one node at a time from temp files. Eliminates the previous in-memory `Cursor<Vec<u8>>` buffer — memory usage is constant regardless of output size.

### Modules

- `main.rs` – CLI (clap), input file collection, `PipelineConfig`, `parse_memory_limit()`, orchestrates the pipeline
- `octree.rs` – `OctreeBuilder`, `RawPoint`, `Bounds`, voxel key assignment, out-of-core temp file management
- `copc_types.rs` – `VoxelKey`, `CopcInfo`, `HierarchyEntry`, VLR/EVLR writing helpers
- `writer.rs` – COPC file writing, streaming LAZ compression, header construction, hierarchy EVLR

### Key constants (in `octree.rs`)

- `MAX_LEAF_POINTS` (65536) – target max points per leaf voxel
- `MAX_NODE_POINTS` (65536) – max points kept per non-leaf (sampled overview)

## Dependencies

- `las` (with `laz-parallel`) – reading LAS/LAZ files via `read_all_points_into` and `read_points_into`
- `laz` – LAZ compression (LasZipCompressor, variable-size chunks)
- `rayon` – parallel file scanning and parallel ancestor node building
- `clap` (derive) – CLI argument parsing
- `byteorder` – little-endian binary I/O
- `anyhow` – error handling
- `log` + `env_logger` – logging
- `tempfile` – temp directory support (though currently uses `std::env::temp_dir` directly)

## Design goals

- Standard-compliant COPC 1.0 output
- Merge multiple input LAZ/LAS files into a single COPC file
- Configurable memory budget via `--memory-limit` with actual disk offloading
- Streaming writer that uses constant memory regardless of output size
- Parallel file scanning and parallel ancestor node building via rayon
- Uses `read_all_points_into` for fast bulk point reading when files fit in budget

## Reference implementations

- [untwine](https://github.com/hobuinc/untwine) – the reference COPC writer
- [LAStools lascopcindex64](https://github.com/LAStools/LAStools) – sometimes generates invalid files
