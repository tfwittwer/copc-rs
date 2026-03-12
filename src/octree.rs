/// Out-of-core octree builder.
///
/// Strategy
/// --------
/// 1. Pass 1 – scan all input files, collect bounding box + point count.
/// 2. Determine octree depth so that leaf nodes contain ≤ MAX_LEAF_POINTS.
/// 3. Pass 2 – read every point, assign it to the leaf voxel key, and
///    accumulate into per-key temporary files on disk.
/// 4. Build the tree bottom-up: each parent node gets a random sample
///    of its children's points (so upper levels are sparse overviews).
/// 5. Produce the list of (VoxelKey, point_count) for the writer.
///
/// Memory usage is bounded by the configurable memory budget. Each pass
/// processes one input file at a time and flushes leaf buffers to disk.
use crate::copc_types::VoxelKey;
use crate::PipelineConfig;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum points per leaf voxel before we subdivide further.
const MAX_LEAF_POINTS: u64 = 65_536;

/// Maximum points to keep per non-leaf node (thinned sample for overview).
const MAX_NODE_POINTS: usize = 65_536;

// ---------------------------------------------------------------------------
// Raw point storage
// ---------------------------------------------------------------------------

/// A raw point stored as scaled integer coordinates plus classification,
/// intensity, return number etc.  We keep the original scaled ints so we
/// can reconstruct exact LAS integer values without floating-point loss.
#[derive(Debug, Clone)]
pub struct RawPoint {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification: u8,
    pub scan_angle: i16,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub nir: u16,
}

impl RawPoint {
    pub const BYTE_SIZE: usize = 4 + 4 + 4 + 2 + 1 + 1 + 1 + 2 + 1 + 2 + 8 + 2 + 2 + 2 + 2; // 38

    pub fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_i32::<LittleEndian>(self.x)?;
        w.write_i32::<LittleEndian>(self.y)?;
        w.write_i32::<LittleEndian>(self.z)?;
        w.write_u16::<LittleEndian>(self.intensity)?;
        w.write_u8(self.return_number)?;
        w.write_u8(self.number_of_returns)?;
        w.write_u8(self.classification)?;
        w.write_i16::<LittleEndian>(self.scan_angle)?;
        w.write_u8(self.user_data)?;
        w.write_u16::<LittleEndian>(self.point_source_id)?;
        w.write_f64::<LittleEndian>(self.gps_time)?;
        w.write_u16::<LittleEndian>(self.red)?;
        w.write_u16::<LittleEndian>(self.green)?;
        w.write_u16::<LittleEndian>(self.blue)?;
        w.write_u16::<LittleEndian>(self.nir)?;
        Ok(())
    }

    pub fn read<R: std::io::Read>(r: &mut R) -> Result<Self> {
        Ok(RawPoint {
            x: r.read_i32::<LittleEndian>()?,
            y: r.read_i32::<LittleEndian>()?,
            z: r.read_i32::<LittleEndian>()?,
            intensity: r.read_u16::<LittleEndian>()?,
            return_number: r.read_u8()?,
            number_of_returns: r.read_u8()?,
            classification: r.read_u8()?,
            scan_angle: r.read_i16::<LittleEndian>()?,
            user_data: r.read_u8()?,
            point_source_id: r.read_u16::<LittleEndian>()?,
            gps_time: r.read_f64::<LittleEndian>()?,
            red: r.read_u16::<LittleEndian>()?,
            green: r.read_u16::<LittleEndian>()?,
            blue: r.read_u16::<LittleEndian>()?,
            nir: r.read_u16::<LittleEndian>()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Bounds {
    pub fn empty() -> Self {
        Bounds {
            min_x: f64::MAX,
            min_y: f64::MAX,
            min_z: f64::MAX,
            max_x: f64::MIN,
            max_y: f64::MIN,
            max_z: f64::MIN,
        }
    }

    pub fn expand_with(&mut self, x: f64, y: f64, z: f64) {
        if x < self.min_x {
            self.min_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if z < self.min_z {
            self.min_z = z;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y > self.max_y {
            self.max_y = y;
        }
        if z > self.max_z {
            self.max_z = z;
        }
    }

    pub fn merge(&mut self, other: &Bounds) {
        if other.min_x < self.min_x {
            self.min_x = other.min_x;
        }
        if other.min_y < self.min_y {
            self.min_y = other.min_y;
        }
        if other.min_z < self.min_z {
            self.min_z = other.min_z;
        }
        if other.max_x > self.max_x {
            self.max_x = other.max_x;
        }
        if other.max_y > self.max_y {
            self.max_y = other.max_y;
        }
        if other.max_z > self.max_z {
            self.max_z = other.max_z;
        }
    }

    /// Cube that contains this AABB.
    pub fn to_cube(&self) -> (f64, f64, f64, f64) {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        let cz = (self.min_z + self.max_z) / 2.0;
        let half = ((self.max_x - self.min_x)
            .max(self.max_y - self.min_y)
            .max(self.max_z - self.min_z))
            / 2.0
            * 1.0001; // tiny epsilon
        (cx, cy, cz, half)
    }
}

// ---------------------------------------------------------------------------
// VoxelKey assignment
// ---------------------------------------------------------------------------

/// Assign a point to the leaf voxel at the given tree depth.
#[allow(clippy::too_many_arguments)]
pub fn point_to_key(
    x: f64,
    y: f64,
    z: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    halfsize: f64,
    depth: u32,
) -> VoxelKey {
    let mut vx = 0i32;
    let mut vy = 0i32;
    let mut vz = 0i32;
    let mut half = halfsize;
    let mut ox = cx;
    let mut oy = cy;
    let mut oz = cz;

    for _ in 0..depth {
        half /= 2.0;
        let bx = if x >= ox {
            vx = vx * 2 + 1;
            ox + half
        } else {
            vx *= 2;
            ox - half
        };
        let by = if y >= oy {
            vy = vy * 2 + 1;
            oy + half
        } else {
            vy *= 2;
            oy - half
        };
        let bz = if z >= oz {
            vz = vz * 2 + 1;
            oz + half
        } else {
            vz *= 2;
            oz - half
        };
        ox = bx;
        oy = by;
        oz = bz;
    }

    VoxelKey {
        level: depth as i32,
        x: vx,
        y: vy,
        z: vz,
    }
}

// ---------------------------------------------------------------------------
// OctreeBuilder
// ---------------------------------------------------------------------------

pub struct OctreeBuilder {
    pub bounds: Bounds,
    pub total_points: u64,
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub halfsize: f64,
    pub depth: u32,
    /// Scale / offset from the first input file – used for all raw-integer conversions.
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    /// Temp directory where node files are written.
    pub tmp_dir: PathBuf,
}

impl OctreeBuilder {
    /// Pass 1: scan all files to get bounds and total point count.
    /// Uses rayon to read file headers in parallel.
    pub fn scan(input_files: &[PathBuf], config: &PipelineConfig) -> Result<Self> {
        type Transforms = (f64, f64, f64, f64, f64, f64);
        let results: Vec<(Bounds, u64, Transforms)> = input_files
            .par_iter()
            .map(|path| -> Result<_> {
                info!("Scanning {:?}", path);
                let reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                let hdr = reader.header();
                let b = hdr.bounds();
                let mut bounds = Bounds::empty();
                bounds.expand_with(b.min.x, b.min.y, b.min.z);
                bounds.expand_with(b.max.x, b.max.y, b.max.z);
                let point_count = hdr.number_of_points();
                let t = hdr.transforms();
                let transforms = (
                    t.x.scale, t.y.scale, t.z.scale, t.x.offset, t.y.offset, t.z.offset,
                );
                Ok((bounds, point_count, transforms))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut bounds = Bounds::empty();
        let mut total_points = 0u64;
        for (b, count, _) in &results {
            bounds.merge(b);
            total_points += count;
        }

        let (scale_x, scale_y, scale_z, offset_x, offset_y, offset_z) = results
            .first()
            .map(|(_, _, t)| *t)
            .unwrap_or((0.001, 0.001, 0.001, 0.0, 0.0, 0.0));

        let (cx, cy, cz, halfsize) = bounds.to_cube();

        // Choose depth so that leaf voxels hold ≤ MAX_LEAF_POINTS on average.
        // Octree has 8^depth leaves.
        let depth = {
            let mut d = 0u32;
            while (total_points as f64) / (8u64.pow(d) as f64) > MAX_LEAF_POINTS as f64 {
                d += 1;
                if d > 16 {
                    break;
                }
            }
            d.max(1)
        };
        info!("Octree depth = {depth}, total points = {total_points}");

        let sys_tmp = std::env::temp_dir();
        let base_tmp = config.temp_dir.as_deref().unwrap_or(&sys_tmp);
        let tmp_dir = base_tmp.join(format!("copc_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir)?;

        Ok(OctreeBuilder {
            bounds,
            total_points,
            cx,
            cy,
            cz,
            halfsize,
            depth,
            scale_x,
            scale_y,
            scale_z,
            offset_x,
            offset_y,
            offset_z,
            tmp_dir,
        })
    }

    /// Path for a node's temp file.
    fn node_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }

    /// Convert a las::Point to a RawPoint using the builder's scale/offset.
    fn convert_point(&self, p: &las::Point) -> RawPoint {
        let ix = ((p.x - self.offset_x) / self.scale_x).round() as i32;
        let iy = ((p.y - self.offset_y) / self.scale_y).round() as i32;
        let iz = ((p.z - self.offset_z) / self.scale_z).round() as i32;

        RawPoint {
            x: ix,
            y: iy,
            z: iz,
            intensity: p.intensity,
            return_number: p.return_number,
            number_of_returns: p.number_of_returns,
            classification: p.classification.into(),
            scan_angle: (p.scan_angle / 0.006).round() as i16,
            user_data: p.user_data,
            point_source_id: p.point_source_id,
            gps_time: p.gps_time.unwrap_or(0.0),
            red: p.color.as_ref().map(|c| c.red).unwrap_or(0),
            green: p.color.as_ref().map(|c| c.green).unwrap_or(0),
            blue: p.color.as_ref().map(|c| c.blue).unwrap_or(0),
            nir: p.extra_bytes.first().copied().map(|_| 0u16).unwrap_or(0),
        }
    }

    /// Parallel key assignment + conversion for a batch of points.
    /// Returns (VoxelKey, RawPoint) pairs computed across all cores.
    fn classify_points_parallel(&self, points: &[las::Point]) -> Vec<(VoxelKey, RawPoint)> {
        points
            .par_iter()
            .map(|p| {
                let key = point_to_key(
                    p.x,
                    p.y,
                    p.z,
                    self.cx,
                    self.cy,
                    self.cz,
                    self.halfsize,
                    self.depth,
                );
                let raw = self.convert_point(p);
                (key, raw)
            })
            .collect()
    }

    /// Merge classified points into per-key buffers and flush periodically.
    fn merge_into_buffers(
        classified: Vec<(VoxelKey, RawPoint)>,
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
        point_idx: &mut u64,
        flush_every: usize,
    ) -> Result<()> {
        for (key, raw) in classified {
            buffers.entry(key).or_default().push(raw);
            *point_idx += 1;
            if (*point_idx).is_multiple_of(flush_every as u64) {
                Self::flush_buffers(buffers, writers, tmp_dir)?;
            }
        }
        Ok(())
    }

    /// Pass 2: assign all points to leaf temp files.
    ///
    /// Uses `read_all_points_into` (fast parallel decompression) when a file
    /// fits in half the memory budget, otherwise falls back to batched
    /// `read_points_into` to stay within bounds.
    ///
    /// Point classification (key assignment + coordinate conversion) is
    /// parallelized across all cores via rayon.
    pub fn distribute(&self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        let flush_every =
            ((config.memory_budget / 4) as usize / RawPoint::BYTE_SIZE).clamp(10_000, 500_000);
        info!("Flush interval: {} points", flush_every);

        let mut buffers: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        let mut writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();
        let mut point_idx = 0u64;

        // Estimated memory per las::Point (~120 bytes)
        let half_budget = config.memory_budget / 2;

        for path in input_files {
            info!("Distributing {:?}", path);
            let mut reader = las::Reader::from_path(path)
                .with_context(|| format!("Cannot open {:?}", path))?;

            let file_point_count = reader.header().number_of_points();
            let estimated_mem = file_point_count * 120;

            if estimated_mem <= half_budget {
                // Fast path: load entire file with parallel decompression
                let mut points: Vec<las::Point> = Vec::new();
                reader.read_all_points_into(&mut points)?;

                // Parallel key assignment + conversion across all cores
                let classified = self.classify_points_parallel(&points);
                drop(points); // free las::Point memory before merging
                Self::merge_into_buffers(
                    classified,
                    &mut buffers,
                    &mut writers,
                    &self.tmp_dir,
                    &mut point_idx,
                    flush_every,
                )?;
            } else {
                // Batched path: read in chunks to stay within budget
                let batch_size = (half_budget / 120).max(10_000);
                info!(
                    "File too large for memory ({} points, ~{} MB), using batched reads of {} points",
                    file_point_count,
                    estimated_mem / (1024 * 1024),
                    batch_size
                );
                let mut points: Vec<las::Point> = Vec::new();
                loop {
                    points.clear();
                    let n = reader.read_points_into(batch_size, &mut points)?;
                    if n == 0 {
                        break;
                    }
                    // Parallel key assignment + conversion for this batch
                    let classified = self.classify_points_parallel(&points);
                    Self::merge_into_buffers(
                        classified,
                        &mut buffers,
                        &mut writers,
                        &self.tmp_dir,
                        &mut point_idx,
                        flush_every,
                    )?;
                }
            }
        }

        // Final flush
        Self::flush_buffers(&mut buffers, &mut writers, &self.tmp_dir)?;
        Ok(())
    }

    fn flush_buffers(
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
    ) -> Result<()> {
        for (key, pts) in buffers.iter_mut() {
            if pts.is_empty() {
                continue;
            }
            let w = writers.entry(*key).or_insert_with(|| {
                let path = tmp_dir.join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z));
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .expect("Cannot open leaf file");
                BufWriter::new(f)
            });
            for p in pts.iter() {
                p.write(w)?;
            }
            pts.clear();
        }
        Ok(())
    }

    /// Read all raw points for a given node key from disk.
    pub fn read_node(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.node_path(key);
        if !path.exists() {
            return Ok(vec![]);
        }
        let f = File::open(&path)?;
        let mut r = BufReader::new(f);
        let file_len = path.metadata()?.len();
        let count = file_len / RawPoint::BYTE_SIZE as u64;
        let mut pts = Vec::with_capacity(count as usize);
        for _ in 0..count {
            pts.push(RawPoint::read(&mut r)?);
        }
        Ok(pts)
    }

    /// Write points to a temp file for the given node key.
    fn write_node_to_temp(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        let path = self.node_path(key);
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        for p in points {
            p.write(&mut w)?;
        }
        Ok(())
    }

    /// Enumerate all leaf keys that have data.
    pub fn leaf_keys(&self) -> Result<Vec<VoxelKey>> {
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(&self.tmp_dir)? {
            let entry = entry?;
            let name = entry.file_name().into_string().unwrap_or_default();
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() == 4
                && let (Ok(l), Ok(x), Ok(y), Ok(z)) = (
                    parts[0].parse::<i32>(),
                    parts[1].parse::<i32>(),
                    parts[2].parse::<i32>(),
                    parts[3].parse::<i32>(),
                )
            {
                keys.push(VoxelKey { level: l, x, y, z });
            }
        }
        Ok(keys)
    }

    /// Build ancestor nodes bottom-up, one level at a time, with disk offloading.
    ///
    /// Returns a list of (VoxelKey, point_count) for all nodes (leaf + ancestor).
    /// Points remain on disk in temp files — the writer reads them on demand.
    pub fn build_node_map(&self) -> Result<Vec<(VoxelKey, usize)>> {
        let leaf_keys = self.leaf_keys()?;
        info!("Number of leaf nodes: {}", leaf_keys.len());

        // Collect all node keys with their point counts
        let mut all_nodes: Vec<(VoxelKey, usize)> = Vec::new();

        // Add leaf nodes
        for key in &leaf_keys {
            let path = self.node_path(key);
            let file_len = path.metadata()?.len();
            let count = file_len as usize / RawPoint::BYTE_SIZE;
            if count > 0 {
                all_nodes.push((*key, count));
            }
        }

        // Build ancestor nodes bottom-up, one level at a time
        let mut current_level_keys: Vec<VoxelKey> =
            leaf_keys.into_iter().filter(|k| {
                let path = self.node_path(k);
                path.metadata().map(|m| m.len() > 0).unwrap_or(false)
            }).collect();

        for d in (0..self.depth).rev() {
            info!("Building ancestor level {d}");

            // Group children at level d+1 by their parent at level d
            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for ck in &current_level_keys {
                if ck.level as u32 == d + 1
                    && let Some(parent) = ck.parent()
                {
                    parent_children.entry(parent).or_default().push(*ck);
                }
            }

            if parent_children.is_empty() {
                continue;
            }

            // Process each parent in parallel with rayon
            let parents: Vec<(VoxelKey, Vec<VoxelKey>)> =
                parent_children.into_iter().collect();

            let new_nodes: Vec<(VoxelKey, usize)> = parents
                .par_iter()
                .map(|(parent, children)| -> Result<(VoxelKey, usize)> {
                    // Read all children in parallel
                    let child_samples: Vec<Vec<RawPoint>> = children
                        .par_iter()
                        .map(|ck| -> Result<Vec<RawPoint>> {
                            let pts = self.read_node(ck)?;
                            Ok(thin_sample(&pts, MAX_NODE_POINTS / 8))
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let mut all_child_points = Vec::new();
                    for sample in child_samples {
                        all_child_points.extend(sample);
                    }
                    let parent_pts = thin_sample(&all_child_points, MAX_NODE_POINTS);
                    let count = parent_pts.len();
                    if count > 0 {
                        self.write_node_to_temp(parent, &parent_pts)?;
                    }
                    Ok((*parent, count))
                })
                .collect::<Result<Vec<_>>>()?;

            let new_keys: Vec<VoxelKey> = new_nodes
                .iter()
                .filter(|(_, c)| *c > 0)
                .map(|(k, _)| *k)
                .collect();

            all_nodes.extend(new_nodes.into_iter().filter(|(_, c)| *c > 0));
            current_level_keys = new_keys;
        }

        info!("Total octree nodes: {}", all_nodes.len());
        Ok(all_nodes)
    }

    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.tmp_dir);
    }
}

/// Sub-sample a point vector to at most `max_count` points using
/// a simple stride-based approach (deterministic, fast).
fn thin_sample(pts: &[RawPoint], max_count: usize) -> Vec<RawPoint> {
    if pts.len() <= max_count {
        return pts.to_vec();
    }
    let step = pts.len() / max_count;
    pts.iter().step_by(step).take(max_count).cloned().collect()
}
