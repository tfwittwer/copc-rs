/// Write a COPC 1.0 file.
///
/// Layout
/// ------
///  [LAS 1.4 header]           375 bytes
///  [copc info VLR]            54 + 160 = 214 bytes
///  [laszip VLR]               54 + 46  = 100 bytes   (user_id "laszip encoded", record_id 22204)
///  --- offset_to_point_data = 689 ---
///  [i64 chunk-table offset]   8 bytes  (points to chunk table after all data)
///  [compressed chunk 0]       variable
///  [compressed chunk 1]       variable
///  ...
///  [LAZ chunk table]          variable (appended after data, referenced by the i64 above)
///  [copc hierarchy EVLR]      60 + n*32 bytes
///
/// The LasZipCompressor writes directly to a BufWriter<File>, eliminating the
/// need for an in-memory buffer. Only one node's points are in memory at a time.
use crate::copc_types::{
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, VoxelKey, write_evlr, write_vlr,
};
use crate::octree::{OctreeBuilder, RawPoint};
use rayon::prelude::*;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use laz::{LasZipCompressor, LazVlrBuilder};
use log::info;
use std::collections::HashSet;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// LAS 1.4 format 7 raw byte size = 30 (base format 6) + 6 (RGB) = 36
// ---------------------------------------------------------------------------
const POINT_RECORD_LENGTH: u16 = 36;

/// Encode one point as LAS 1.4 format 7 raw bytes (36 bytes, little-endian).
fn encode_point_fmt7(rp: &RawPoint, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&rp.x.to_le_bytes()); // 4  X (scaled integer)
    buf.extend_from_slice(&rp.y.to_le_bytes()); // 4  Y
    buf.extend_from_slice(&rp.z.to_le_bytes()); // 4  Z
    buf.extend_from_slice(&rp.intensity.to_le_bytes()); // 2  Intensity
    // Return bit fields: bits 0-3 = return number, bits 4-7 = number of returns
    let return_byte = (rp.return_number & 0x0F) | ((rp.number_of_returns & 0x0F) << 4);
    buf.push(return_byte); // 1
    buf.push(0u8); // classification flags / scanner channel / scan dir / edge
    buf.push(rp.classification); // 1  Classification
    buf.push(rp.user_data); // 1  User data
    buf.extend_from_slice(&rp.scan_angle.to_le_bytes()); // 2  Scan angle (0.006° units, i16)
    buf.extend_from_slice(&rp.point_source_id.to_le_bytes()); // 2  Point source ID
    buf.extend_from_slice(&rp.gps_time.to_le_bytes()); // 8  GPS time
    buf.extend_from_slice(&rp.red.to_le_bytes()); // 2  Red
    buf.extend_from_slice(&rp.green.to_le_bytes()); // 2  Green
    buf.extend_from_slice(&rp.blue.to_le_bytes()); // 2  Blue
    // Total = 4+4+4+2+1+1+1+1+2+2+8+2+2+2 = 36 bytes ✓
}

/// Write a complete COPC file to `output_path`.
///
/// Streams compressed data directly to the output file — only one node's
/// points are in memory at a time (~2.5 MB max).
pub fn write_copc(
    output_path: &Path,
    builder: &OctreeBuilder,
    node_keys: &[(VoxelKey, usize)],
) -> Result<()> {
    // -----------------------------------------------------------------------
    // Scale / offset – taken from the builder (captured from the first input file)
    // -----------------------------------------------------------------------
    let scale_x = builder.scale_x;
    let scale_y = builder.scale_y;
    let scale_z = builder.scale_z;
    let offset_x = builder.offset_x;
    let offset_y = builder.offset_y;
    let offset_z = builder.offset_z;

    // -----------------------------------------------------------------------
    // Build the LAZ VLR (variable-size chunks, format 7 items)
    // -----------------------------------------------------------------------
    let laz_vlr = LazVlrBuilder::default()
        .with_point_format(7, 0)
        .context("LazVlrBuilder for format 7")?
        .with_variable_chunk_size()
        .build();

    let mut laz_vlr_payload: Vec<u8> = Vec::new();
    laz_vlr.write_to(&mut laz_vlr_payload)?;

    // -----------------------------------------------------------------------
    // File layout constants
    // -----------------------------------------------------------------------
    let copc_info_vlr_size: u32 = 54 + 160; // 214
    let laz_vlr_size: u32 = 54 + laz_vlr_payload.len() as u32; // 100
    let offset_to_point_data: u32 = 375 + copc_info_vlr_size + laz_vlr_size;

    let copc_info_payload_pos: u64 = 375 + 54; // byte position of copc info payload in file

    let b = &builder.bounds;

    // -----------------------------------------------------------------------
    // Build BFS-ordered key list from node_keys
    // -----------------------------------------------------------------------
    let key_set: HashSet<VoxelKey> = node_keys.iter().map(|(k, _)| *k).collect();
    let point_counts: std::collections::HashMap<VoxelKey, usize> =
        node_keys.iter().copied().collect();

    let ordered_keys: Vec<VoxelKey> = {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(VoxelKey::root());
        while let Some(k) = queue.pop_front() {
            if key_set.contains(&k) {
                result.push(k);
                for child in k.children() {
                    queue.push_back(child);
                }
            }
        }
        result
    };

    // -----------------------------------------------------------------------
    // Write LAS 1.4 header manually (375 bytes)
    // -----------------------------------------------------------------------
    let file = std::fs::File::create(output_path)
        .with_context(|| format!("Cannot create {:?}", output_path))?;
    let mut w = BufWriter::new(file);

    w.write_all(b"LASF")?;
    w.write_u16::<LittleEndian>(0)?; // file source ID
    w.write_u16::<LittleEndian>(0x0001 | 0x0010)?; // global encoding: GPS standard + WKT
    w.write_all(&[0u8; 16])?; // project ID (GUID)
    w.write_u8(1)?; // version major
    w.write_u8(4)?; // version minor
    let mut sysid = [0u8; 32];
    b"copc_converter"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| sysid[i] = c);
    w.write_all(&sysid)?; // system identifier
    let mut gensoft = [0u8; 32];
    b"copc_converter 0.1"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| gensoft[i] = c);
    w.write_all(&gensoft)?; // generating software
    w.write_u16::<LittleEndian>(1)?; // file creation day
    w.write_u16::<LittleEndian>(2024)?; // file creation year
    w.write_u16::<LittleEndian>(375)?; // header size
    w.write_u32::<LittleEndian>(offset_to_point_data)?; // offset to point data
    w.write_u32::<LittleEndian>(2)?; // number of VLRs (copc info + laszip)
    w.write_u8(128 | 7)?; // point data format: 135 = LAZ-compressed format 7
    w.write_u16::<LittleEndian>(POINT_RECORD_LENGTH)?; // point data record length = 36
    w.write_u32::<LittleEndian>(0)?; // legacy point count
    for _ in 0..5 {
        w.write_u32::<LittleEndian>(0)?;
    } // legacy point counts by return
    w.write_f64::<LittleEndian>(scale_x)?;
    w.write_f64::<LittleEndian>(scale_y)?;
    w.write_f64::<LittleEndian>(scale_z)?;
    w.write_f64::<LittleEndian>(offset_x)?;
    w.write_f64::<LittleEndian>(offset_y)?;
    w.write_f64::<LittleEndian>(offset_z)?;
    w.write_f64::<LittleEndian>(b.max_x)?;
    w.write_f64::<LittleEndian>(b.min_x)?;
    w.write_f64::<LittleEndian>(b.max_y)?;
    w.write_f64::<LittleEndian>(b.min_y)?;
    w.write_f64::<LittleEndian>(b.max_z)?;
    w.write_f64::<LittleEndian>(b.min_z)?;
    w.write_u64::<LittleEndian>(0)?; // start of waveform data
    // start_of_first_EVLR (u64) at byte offset 235 – placeholder, patched below
    w.write_u64::<LittleEndian>(0)?;
    w.write_u32::<LittleEndian>(1)?; // number of EVLRs
    w.write_u64::<LittleEndian>(builder.total_points)?; // point count (u64)
    for _ in 0..15 {
        w.write_u64::<LittleEndian>(0)?;
    } // point counts by return

    // -----------------------------------------------------------------------
    // VLR 1: copc info (placeholder – patched at the end)
    // -----------------------------------------------------------------------
    let copc_info_placeholder = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
        root_hier_offset: 0,
        root_hier_size: 0,
        gpstime_minimum: 0.0,
        gpstime_maximum: 0.0,
    };
    let mut copc_info_buf = Vec::with_capacity(160);
    copc_info_placeholder.write(&mut copc_info_buf)?;
    write_vlr(&mut w, "copc", 1, "copc info", &copc_info_buf)?;

    // -----------------------------------------------------------------------
    // VLR 2: laszip VLR
    // -----------------------------------------------------------------------
    write_vlr(
        &mut w,
        "laszip encoded",
        22204,
        "laz variable chunks",
        &laz_vlr_payload,
    )?;

    // Flush header+VLRs so the file is complete up to offset_to_point_data.
    w.flush()?;

    // -----------------------------------------------------------------------
    // Stream compressed point data directly to file via LasZipCompressor
    // -----------------------------------------------------------------------
    let laz_vlr_for_compressor = LazVlrBuilder::default()
        .with_point_format(7, 0)
        .context("LazVlrBuilder (compressor)")?
        .with_variable_chunk_size()
        .build();

    // The compressor writes directly to the output file. stream_position()
    // returns absolute file positions, so done() patches the chunk-table
    // offset correctly without manual arithmetic.
    let mut compressor = LasZipCompressor::new(w, laz_vlr_for_compressor)
        .map_err(|e| anyhow::anyhow!("LasZipCompressor::new: {e}"))?;

    compressor
        .reserve_offset_to_chunk_table()
        .context("reserve_offset_to_chunk_table")?;

    let mut chunk_starts: Vec<u64> = Vec::with_capacity(ordered_keys.len());

    // Process nodes in batches: read + encode in parallel, compress sequentially.
    // The compressor is stateful so compression must be serial, but reading from
    // disk and encoding to format-7 bytes are embarrassingly parallel.
    let batch_size = rayon::current_num_threads().max(4);
    let total_keys = ordered_keys.len();
    let mut global_idx = 0usize;

    for batch in ordered_keys.chunks(batch_size) {
        // Parallel read from disk + encode to format-7 bytes
        let encoded_batch: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|key| -> Result<Vec<u8>> {
                let pts = builder.read_node(key)?;
                let mut raw_bytes =
                    Vec::with_capacity(POINT_RECORD_LENGTH as usize * pts.len());
                for rp in &pts {
                    encode_point_fmt7(rp, &mut raw_bytes);
                }
                Ok(raw_bytes)
            })
            .collect::<Result<Vec<_>>>()?;

        // Sequential compression (compressor is stateful)
        for raw_bytes in &encoded_batch {
            let chunk_start = compressor.get_mut().stream_position()?;
            chunk_starts.push(chunk_start);

            compressor
                .compress_many(raw_bytes)
                .context("compress_many")?;

            if global_idx < total_keys - 1 {
                compressor
                    .finish_current_chunk()
                    .context("finish_current_chunk")?;
            }
            global_idx += 1;
        }
    }

    // Finalize: writes the last chunk and the chunk table, patches the offset.
    compressor.done().context("compressor done")?;

    let mut w = compressor.into_inner();
    let end_pos = w.stream_position()?;
    w.flush()?;

    // Get the underlying File so we can read back the chunk-table offset
    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("BufWriter flush: {}", e.error()))?;

    // Read the chunk-table offset written by the compressor at offset_to_point_data
    file.seek(SeekFrom::Start(offset_to_point_data as u64))?;
    let chunk_table_pos = file.read_i64::<LittleEndian>()? as u64;

    // -----------------------------------------------------------------------
    // Build chunk_info for the hierarchy EVLR
    // -----------------------------------------------------------------------
    let mut chunk_info: Vec<(VoxelKey, u64, i32, i32)> = Vec::new();

    for (i, key) in ordered_keys.iter().enumerate() {
        let pc = point_counts.get(key).copied().unwrap_or(0);
        let cursor_start = chunk_starts[i];
        let cursor_end = if i + 1 < chunk_starts.len() {
            chunk_starts[i + 1]
        } else {
            chunk_table_pos
        };
        let byte_size = (cursor_end - cursor_start) as i32;
        chunk_info.push((*key, cursor_start, byte_size, pc as i32));
        info!(
            "Chunk {:?}: offset={}, size={}, pts={}",
            key, cursor_start, byte_size, pc
        );
    }

    // -----------------------------------------------------------------------
    // EVLR: copc hierarchy
    // -----------------------------------------------------------------------
    let evlr_start = end_pos;

    let mut hier_payload: Vec<u8> = Vec::with_capacity(chunk_info.len() * 32);
    for (key, offset, byte_size, point_count) in &chunk_info {
        HierarchyEntry {
            key: *key,
            offset: *offset,
            byte_size: *byte_size,
            point_count: *point_count,
        }
        .write(&mut hier_payload)?;
    }

    // Seek to end of compressed data to write the EVLR
    file.seek(SeekFrom::Start(evlr_start))?;
    let mut w = BufWriter::new(file);
    write_evlr(&mut w, "copc", 1000, "copc hierarchy", &hier_payload)?;
    w.flush()?;
    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("BufWriter flush: {}", e.error()))?;

    // -----------------------------------------------------------------------
    // Patch the file: copc info VLR + EVLR offset
    // -----------------------------------------------------------------------

    // Patch copc info: root_hier_offset / root_hier_size
    let patched_info = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
        root_hier_offset: evlr_start + EVLR_HEADER_SIZE as u64,
        root_hier_size: hier_payload.len() as u64,
        gpstime_minimum: 0.0,
        gpstime_maximum: 0.0,
    };
    let mut pinfo_buf = Vec::with_capacity(160);
    patched_info.write(&mut pinfo_buf)?;
    file.seek(SeekFrom::Start(copc_info_payload_pos))?;
    file.write_all(&pinfo_buf)?;

    // Patch start_of_first_EVLR at byte 235
    file.seek(SeekFrom::Start(235))?;
    file.write_all(&evlr_start.to_le_bytes())?;

    info!("COPC file written: {:?}", output_path);
    Ok(())
}
