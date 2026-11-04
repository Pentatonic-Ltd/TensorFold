//! Opt-in shared verify capacity. Keep every admitted window within one target pass.
const std = @import("std");

pub const Capacity = struct {
    shared: bool = false,
    rows: usize,
    streams: u32,
};

pub fn wanted() bool {
    const v = std.c.getenv("TF_FLASHNEXT_RAGGED_VERIFY") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Reserve for worst-case windows, including copied continuations, before streams are admitted.
/// Small chunks and two ranks retain the existing path.
pub fn capacity(on: bool, ranked: bool, chunk: usize, window: usize, legacy: usize) Capacity {
    if (!on or ranked or chunk < 2 * window) return .{ .rows = legacy, .streams = 32 };
    const rows = @min(chunk, 32 * window);
    return .{ .shared = true, .rows = rows, .streams = @intCast(rows / window) };
}

test "ragged verify capacity: every admitted window fits one pass" {
    for ([_]usize{ 64, 65, 159, 160, 256, 1024, 1536, 16384 }) |chunk| {
        const c = capacity(true, false, chunk, 32, 64);
        try std.testing.expect(c.rows <= chunk);
        try std.testing.expect(c.streams * 32 <= c.rows);
        try std.testing.expect(c.streams >= 2 and c.streams <= 32);
    }
    const five = capacity(true, false, 160, 32, 64);
    try std.testing.expectEqual(@as(u32, 5), five.streams);
    for ([_]Capacity{ capacity(false, false, 1536, 32, 64), capacity(true, true, 1536, 32, 64), capacity(true, false, 32, 32, 64) }) |c| {
        try std.testing.expectEqual(@as(usize, 64), c.rows);
        try std.testing.expectEqual(@as(u32, 32), c.streams);
    }
}
